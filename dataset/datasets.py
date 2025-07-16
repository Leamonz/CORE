import os
import h5py
import numpy as np
from torch.utils.data import Dataset

dataset_registry = {
    "carbon_tracker": lambda args: CarbonTrackerDataset(args),
    "cmip6": lambda args: CMIP6Dataset(args)
}

def build_dataset(args, data_type):
    if data_type not in dataset_registry:
        raise ValueError(f"Unknown dataset type: {data_type}")
    return dataset_registry[data_type](args)

class CarbonTrackerDataset(Dataset):
    """
    read dataset from h5 files
    """

    def __init__(self, data_args):
        self.args = data_args
        if self.args.resolution is None:
            self.resolution = 1
        else:
            self.resolution = self.args.resolution

        image_size = self.args.image_size
        self.h5_path = self.args.data_path
        with h5py.File(self.h5_path, "r") as f:
            self.dataset_keys = list(f.keys())  # get dataset keys
            self.length = f[self.args.input_vars[-1]].shape[0]
            self.length = self.length // self.resolution - self.args.num_frames

            self.means = {var: f["mean"][var][()] for var in f["mean"]}
            self.stds = {var: f["std"][var][()] for var in f["std"]}

        ## Timestamp for embedding
        time_steps = self.args.num_frames
        num_samples_per_year = math.ceil(365 * (24 / self.args.hrs)) / self.resolution
        single_year = num_samples_per_year - time_steps
        nyears = self.data.shape[0] // single_year
        self.time = []
        for i in range(nyears):
            time_idx = list(range(num_samples_per_year))
            self.time += time_idx[time_steps ::]

    def __len__(self):
        return self.length

    def normalize(self, data, var_name, lsm, eps=1e-6):
        # only normalize on ocean areas
        mean, std = self.means[var_name][()], self.stds[var_name][()]
        normalized_data = np.where(
            lsm == 1.0, (data - np.asarray(mean)) / (np.asarray(std) + eps), data
        )
        return normalized_data

    def __getitem__(self, index):
        index = index * self.resolution
        input_data = []
        gt = []
        with h5py.File(self.h5_path, "r") as f:
            ## input variables
            for var in self.args.input_vars:
                if "observation" in var:
                    var_data = (
                        f[var]
                        if not self.args.sampling_ratio
                        else f[var]["%.1f" % self.args.sampling_ratio]
                    )
                    var_name = var.split("_")[0]
                else:
                    var_data = f[var]
                    var_name = var

                data_chunk = var_data[
                    index : (
                        index + self.args.num_frames * self.resolution
                    ) : self.resolution,
                    0:1,
                ]  # (t, 1, h, w)
                data_chunk = np.nan_to_num(
                    data_chunk, nan=0.0
                )  # in case nans are on land
                input_data.append(self.normalize(data_chunk, var_name, lsm))

            ## output variables
            for var in self.args.output_vars:
                var_name = var
                data_chunk = f[var][
                    index : (
                        index + self.args.num_frames * self.resolution
                    ) : self.resolution,
                    0:1,
                ]  # (t, 1, h, w)
                data_chunk = np.nan_to_num(data_chunk, nan=0.0)
                gt.append(self.normalize(data_chunk, var_name, lsm))

        input_data = np.concatenate(input_data, axis=1)
        gt = np.concatenate(gt, axis=1)
        t = self.time[index]

        input_data = torch.from_numpy(input_data)
        gt = torch.from_numpy(gt)
        t = torch.from_numpy(t)

        return input_data, gt, t


class CMIP6Dataset(Dataset):
    def __init__(self, data_args):
        self.args = data_args
        if self.args.resolution is None:
            self.resolution = 1
        else:
            self.resolution = self.args.resolution

        image_size = self.args.image_size
        self.h5_path = self.args.data_path
        with h5py.File(self.h5_path, "r") as f:
            self.dataset_keys = list(f.keys())  # get dataset keys
            self.length = f[self.args.input_vars[-1]].shape[0]
            self.length = self.length // self.resolution - self.args.num_frames

            self.means = {var: f["mean"][var][()] for var in f["mean"]}
            self.stds = {var: f["std"][var][()] for var in f["std"]}

        ## Timestamp for embedding
        time_steps = self.args.num_frames
        single_year = 12
        nyears = self.data.shape[0] // single_year
        self.time = []
        for i in range(nyears):
            time_idx = list(range(12))
            self.time += time_idx[time_steps :]

    def __len__(self):
        return self.length

    def normalize(self, data, var_name, lsm, eps=1e-6):
        # only normalize on ocean areas
        mean, std = self.means[var_name][()], self.stds[var_name][()]
        normalized_data = np.where(
            lsm == 1.0, (data - np.asarray(mean)) / (np.asarray(std) + eps), data
        )
        return normalized_data

    def __getitem__(self, index):
        index = index * self.resolution
        input_data = []
        gt = []
        with h5py.File(self.h5_path, "r") as f:
            ## input variables
            for var in self.args.input_vars:
                if "observation" in var:
                    var_data = (
                        f[var]
                        if not self.args.sampling_ratio
                        else f[var]["%.1f" % self.args.sampling_ratio]
                    )
                    var_name = var.split("_")[0]
                else:
                    var_data = f[var]
                    var_name = var

                data_chunk = var_data[
                    index : (
                        index + self.args.num_frames * self.resolution
                    ) : self.resolution,
                    0:1,
                ]  # (t, 1, h, w)
                data_chunk = np.nan_to_num(
                    data_chunk, nan=0.0
                )  # in case nans are on land
                input_data.append(self.normalize(data_chunk, var_name, lsm))

            ## output variables
            for var in self.args.output_vars:
                var_name = var
                data_chunk = f[var][
                    index : (
                        index + self.args.num_frames * self.resolution
                    ) : self.resolution,
                    0:1,
                ]  # (t, 1, h, w)
                data_chunk = np.nan_to_num(data_chunk, nan=0.0)
                gt.append(self.normalize(data_chunk, var_name, lsm))

        input_data = np.concatenate(input_data, axis=1)
        gt = np.concatenate(gt, axis=1)
        t = self.time[index : index + self.args.num_frames]

        input_data = torch.from_numpy(input_data)
        gt = torch.from_numpy(gt)
        t = torch.from_numpy(t)

        return input_data, gt, t
        
