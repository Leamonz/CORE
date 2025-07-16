import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os
import hydra

from omegaconf import DictConfig, OmegaConf
from pprint import pprint
from functools import partial
from pathlib import Path
from collections import OrderedDict

from timm.models import create_model
from timm.utils import ModelEma
from util_tools.optim_factory import (
    create_optimizer,
    LayerDecayValueAssigner,
)

from dataset.datasets import build_dataset
from util_tools.utils import (
    load_bidir_weights,
    unfreeze_block,
)
from util_tools.utils import cross_multiple_samples_collate, laod_eval_weights
import util_tools.utils as utils
from models.model_registry import *
from functions import train_one_epoch, validation_one_epoch, final_test

import sys
import wandb
import pandas as pd


@hydra.main(config_path="./configs", config_name="config")
def main(
    cfg: DictConfig,
):
    print(DictConfig)
    # 从配置中读取 DDP（分布式数据并行）相关配置项
    ddp = cfg.ddp

    # 初始化分布式训练模式，设置 rank、world size、通信 backend 等
    utils.init_distributed_mode(ddp)

    # 获取运行目录和日志目录的路径（用于保存模型权重、日志等）
    run_dir = cfg.dir
    log_dir = cfg.log_dir

    # 如果路径不存在则创建运行目录和日志目录
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 美化打印整个配置，方便调试和确认配置项
    pprint(cfg)

    device = torch.device("cuda")
    # Set up seed manually
    cfg.seed = cfg.seed + utils.get_rank()
    utils.manual_seed(cfg.seed)
    cudnn.benchmark = True

    ###########################
    dataset_val = None
    dataset_test = None

    # loading dataset
    train_data_args, val_data_args, test_data_args = cfg.train_data, cfg.val_data, cfg.test_data
    dataset_val = build_dataset(val_data_args, cfg.data_type)
    dataset_test = build_dataset(test_data_args, cfg.data_type)
    dataset_train = build_dataset(train_data_args, cfg.data_type)

    # detect distribution
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    if cfg.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print(
                "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )
        sampler_val = (
            torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
            if dataset_val
            else None
        )
        sampler_test = (
            torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
            if dataset_test
            else None
        )
    else:
        sampler_val = (
            torch.utils.data.SequentialSampler(dataset_val) if dataset_val else None
        )
        sampler_test = (
            torch.utils.data.SequentialSampler(dataset_test) if dataset_test else None
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=train_data_args.batch_size,
        num_workers=train_data_args.num_workers,
        pin_memory=train_data_args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=5 * val_data_args.batch_size,
            num_workers=val_data_args.num_workers,
            pin_memory=val_data_args.pin_mem,
            drop_last=False,
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=5 * test_data_args.batch_size,
            num_workers=test_data_args.num_workers,
            pin_memory=test_data_args.pin_mem,
            drop_last=False,
        )
    else:
        data_loader_test = None

    total_batch_size = train_data_args.batch_size * cfg.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    # scale lr according to batch_size
    # stablize training process
    optim_args = cfg.optim
    optim_args.lr = optim_args.lr * total_batch_size / 256
    optim_args.min_lr = optim_args.min_lr * total_batch_size / 256
    optim_args.warmup_lr = optim_args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % optim_args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % cfg.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

    # model configuration
    model = create_model(
        cfg.model_name,
        pretrained=False,
        **cfg.model
    )

    model.to(device)

    model_ema = None
    if cfg.model_ema:
        model_ema = ModelEma(
            model,
            decay=cfg.model_ema_decay,
            device="cpu" if cfg.model_ema_force_cpu else "",
            resume="",
        )
        print("Using EMA with decay = %.8f" % cfg.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params:", n_parameters)

    num_layers = model_without_ddp.get_num_layers()
    if cfg.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(
                cfg.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)
            )
        )
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    optimizer = create_optimizer(
        optim_args,
        model_without_ddp,
        skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None,
    )

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        optim_args.lr,
        optim_args.min_lr,
        optim_args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=optim_args.warmup_epochs,
        warmup_steps=optim_args.warmup_steps,
        start_warmup_value=optim_args.warmup_lr,
    )
    if optim_args.weight_decay_end is None:
        optim_args.weight_decay_end = optim_args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        optim_args.weight_decay,
        optim_args.weight_decay_end,
        optim_args.epochs,
        num_training_steps_per_epoch,
    )
    print(
        "Max WD = %.7f, Min WD = %.7f"
        % (max(wd_schedule_values), min(wd_schedule_values))
    )

    if ddp.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[ddp.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    utils.auto_load_model(
        args=cfg,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        model_ema=model_ema,
    )

    # criterion = LatitudeWeightedMSELoss() if args.weighted_loss else nn.MSELoss()
    criterion = nn.MSELoss()
    print("criterion = %s" % str(criterion))
    print(f"Start training for {cfg.epochs} epochs")
    start_time = time.time()
    torch.cuda.empty_cache()
    best_loss = 1e9

    val_epochs = []
    val_rmses = []
    val_cont_losses = []

    train_losses, val_losses, test_losses = [], [], []
    wind_losses = []

    for epoch in range(cfg.start_epoch, cfg.epochs):

        if ddp.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # train
        train_stats = train_one_epoch(
            cfg,
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            cfg.clip_grad,
            model_ema,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=cfg.update_freq,
        )
        torch.cuda.empty_cache()

        # validation & test
        if (
            cfg.valid_freq > 0 and (epoch + 1) % cfg.valid_freq == 0
        ) or epoch + 1 == cfg.epochs:
            val_stats = validation_one_epoch(
                cfg, data_loader_val, model, device, criterion
            )
            test_stats, _ = final_test(
                cfg, data_loader_test, model, device, criterion, file=None
            )
            val_losses.append(val_stats["loss"])

            val_loss = val_stats["rmse"]
            if cfg.save_ckpt and val_loss < best_loss:
                best_loss = val_loss
                print(f"Saving Model for Lower Validation Loss: {val_loss}.")
                utils.save_model(
                    args=cfg,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    epoch=epoch,
                    model_ema=model_ema,
                )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
            val_epochs.append(epoch + 1)
            val_rmses.append(val_stats["rmse"])
            val_cont_losses.append(val_stats["cont_loss"])
        if (epoch + 1) % cfg.save_ckpt_freq == 0:
            utils.save_model(
                args=cfg,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                model_ema=model_ema,
            )

    val_df = pd.DataFrame(
        {
            "epochs": val_epochs,
            "val_rmses": val_rmses,
            "val_cont_losses": val_cont_losses,
        }
    )
    val_df.to_csv(os.path.join(cfg.output_dir, "val_df.csv"), index=False)

    preds_file = os.path.join(cfg.output_dir, str(global_rank) + ".txt")
    test_stats, _ = final_test(
        cfg, data_loader_test, model, device, criterion, file=None
    )
    if cfg.distributed:
        torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    # opts = get_args()
    main()
