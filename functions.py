import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from util_tools.mixup import Mixup
from timm.utils import accuracy, ModelEma
import util_tools.utils as utils
from scipy.special import softmax
from einops import rearrange
from util_tools.metrics import (
    weighted_rmse_torch,
    weighted_rmse_torch_channels,
    weighted_rmse_torch_channels_masked,
    weighted_acc_torch,
    weighted_acc_torch_channels,
    weighted_acc_torch_channels_masked,
)


def cross_train_class_batch(model, samples, target, time, criterion):
    outputs = model(samples, time)
    if isinstance(outputs, tuple):
        output, cont_loss = outputs
    else:
        output, cont_loss = outputs, None

    loss = criterion(output, target) + cont_loss
    return loss, output


def train_one_epoch(
    args,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    model.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if (
            lr_schedule_values is not None
            or wd_schedule_values is not None
            and data_iter_step % update_freq == 0
        ):
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = batch[0]
        targets = batch[1]
        time = batch[2]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        time = time.to(device, non_blocking=True)

        # loss, output = cross_train_class_batch(model, samples, targets, time, criterion)
        outputs = model(samples, time)
        if isinstance(outputs, tuple):
            output, cont_loss = outputs
        else:
            output, cont_loss = outputs, None
        if cont_loss is not None:
            loss = criterion(output, targets) + args.loss_alpha * cont_loss
            metric_logger.update(cont_loss=cont_loss.item())
        else:
            loss = criterion(output, targets)
            metric_logger.update(cont_loss=0.0)

        loss /= update_freq
        loss.backward()
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.step()
            optimizer.zero_grad()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        # metric_logger.update(grad_norm=grad_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.median for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(args, data_loader, model, device, criterion):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val:"

    # switch to evaluation mode
    model.eval()
    running_loss = 0.0
    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch[0]
        targets = batch[1]
        time = batch[2]
        batch_size = samples.shape[0]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        time = time.to(device, non_blocking=True)

        # compute output
        outputs = model(samples, time)
        if isinstance(outputs, tuple):
            output, cont_loss = outputs
        else:
            output, cont_loss = outputs, None
        if cont_loss is not None:
            loss = criterion(output, targets) + args.loss_alpha * cont_loss
            metric_logger.update(cont_loss=cont_loss.item())
        else:
            loss = criterion(output, targets)
            metric_logger.update(cont_loss=0.0)
        rmse = weighted_rmse_torch(output, targets)

        metric_logger.update(rmse=rmse.mean().item())
        metric_logger.update(loss=loss.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Validation Averaged stats:", metric_logger)

    return {k: meter.median for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(args, data_loader, model, device, criterion, file):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()
    final_result = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch[0]
        targets = batch[1]
        time = batch[2]
        batch_size = samples.shape[0]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        time = time.to(device, non_blocking=True)

        # compute output
        outputs = model(samples, time)
        if isinstance(outputs, tuple):
            output, cont_loss = outputs
        else:
            output, cont_loss = outputs, None

        if cont_loss is not None:
            loss = criterion(output, targets) + args.loss_alpha * cont_loss
        else:
            loss = criterion(output, targets)
        rmse = weighted_rmse_torch(output, targets)
        final_result += [r.item() for r in rmse]

        metric_logger.update(loss=loss.item())
        metric_logger.update(rmse=rmse.mean().item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Test Averaged Stats:", metric_logger)
    final_rmse = np.mean(final_result)

    return {k: meter.median for k, meter in metric_logger.meters.items()}, final_rmse


# def merge(eval_path, num_tasks):
#     dict_feats = {}
#     dict_label = {}
#     dict_pos = {}
#     print("Reading individual output files")

#     for x in range(num_tasks):
#         file = os.path.join(eval_path, str(x) + ".txt")
#         lines = open(file, "r").readlines()[1:]
#         for line in lines:
#             line = line.strip()
#             name = line.split("[")[0]
#             label = line.split("]")[1].split(" ")[1]
#             chunk_nb = line.split("]")[1].split(" ")[2]
#             split_nb = line.split("]")[1].split(" ")[3]
#             data = np.fromstring(
#                 line.split("[")[1].split("]")[0], dtype=np.float, sep=","
#             )
#             data = softmax(data)
#             if not name in dict_feats:
#                 dict_feats[name] = []
#                 dict_label[name] = 0
#                 dict_pos[name] = []
#             if chunk_nb + split_nb in dict_pos[name]:
#                 continue
#             dict_feats[name].append(data)
#             dict_pos[name].append(chunk_nb + split_nb)
#             dict_label[name] = label
#     print("Computing final results")

#     input_lst = []
#     print(len(dict_feats))
#     for i, item in enumerate(dict_feats):
#         input_lst.append([i, item, dict_feats[item], dict_label[item]])
#     from multiprocessing import Pool

#     p = Pool(64)
#     ans = p.map(compute_video, input_lst)
#     top1 = [x[1] for x in ans]
#     top5 = [x[2] for x in ans]
#     pred = [x[0] for x in ans]
#     label = [x[3] for x in ans]
#     final_top1, final_top5 = np.mean(top1), np.mean(top5)
#     return final_top1 * 100, final_top5 * 100


# def compute_video(lst):
#     i, video_id, data, label = lst
#     feat = [x for x in data]
#     feat = np.mean(feat, axis=0)
#     pred = np.argmax(feat)
#     top1 = (int(pred) == int(label)) * 1.0
#     top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
#     return [pred, top1, top5, int(label)]
