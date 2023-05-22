# The MIT License (MIT)
#
# Copyright (c) 2022-2023, Mohsen Sadeghi (mohsen.sadeghi@fu-berlin)
# Artificial Intelligence for the Sciences Group (AI4Science),
# Freie Universität Berlin, Germany.
#
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import numpy as np
import pickle
from . import network_components as nc

channel_wavelength = [405.0, 405.0, 405.0, 405.0, 488.0, 488.0, 488.0, 561.0, 561.0, 633.0]
n_channels = len(channel_wavelength)


class IterableDatasetAugmentor(torch.utils.data.IterableDataset):
    def __init__(self, dataset, augment=False, brightness_scaling_factor=0.0):
        super(IterableDatasetAugmentor).__init__()

        self.dataset = dataset

        self.brightness_scaling_factor = brightness_scaling_factor

        if augment:
            self.transform = self.augment_brightness
        else:
            self.transform = self.do_not_augment

        self.start = 0
        self.end = len(self.dataset)

    def do_not_augment(self, x):
        return x

    def augment_brightness(self, x):
        return x * (1.0 + (2.0 * torch.rand(size=x.size(), device=nc.nn_device) - 1.0) * self.brightness_scaling_factor)

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(torch.math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        _dat = self.dataset[iter_start: iter_end]

        return zip(map(self.transform, _dat[0]), _dat[1])

    def __len__(self):

        return len(self.dataset)


class DatasetAugmentor(torch.utils.data.Dataset):

    def __init__(self, dataset, augment=False, brightness_scaling_factor=0.0):

        super(DatasetAugmentor, self).__init__()

        self.dataset = dataset

        self.transform = lambda x: x

        if augment:
            self.transform = lambda x: x * (
                    1.0 + (torch.rand(size=x.size(), device=nc.nn_device) - 0.5) * brightness_scaling_factor)

        _sum = torch.zeros(n_channels).to(nc.nn_device)
        _sum2 = torch.zeros(n_channels).to(nc.nn_device)
        _n = 0.0

        for data, target in self.dataset:
            _sum += data
            _sum2 += data ** 2
            _n += 1.0

        self.dataset_mean = _sum / _n
        self.dataset_std = torch.sqrt(_sum2 / _n - self.dataset_mean ** 2)

        print(f"Mean value of the input data = {self.dataset_mean}")
        print(f"Standard deviation of the input data = {self.dataset_std}")

        data, target = self.dataset[0]
        print(f"Example of augmented data = {self.transform(data), target}")

    def __getitem__(self, index):

        brightness_data, target = self.dataset[index]

        return self.transform(brightness_data), target

    # def __getitems__(self, indices):
    #
    #     _dat = self.dataset[indices]
    #
    #     return list(zip(map(self.transform, _dat[0]), _dat[1]))

    def __len__(self):

        return len(self.dataset)


class NanobarcodeDataset:

    def __init__(self, brightness_data_file_name,
                 train_val_test_split_frac=(0.8, 0.1, 0.1),
                 do_brightness_augmentation=False,
                 brightness_augmentation_factor=0.3,
                 verbose=False):

        # A fixed random seed to make the dataset splitting reproducible
        self.random = np.random.default_rng(23847)

        print(f"loading data from {brightness_data_file_name}...")

        self.brightness_data, self.protein_names, \
            self.channel_mean, self.channel_std = self.load_brightness_data(brightness_data_file_name, verbose)

        self.id_to_protein_name = [""] * len(self.protein_names)

        for _pr in self.protein_names:
            self.id_to_protein_name[self.brightness_data[_pr]["ID"]] = _pr

        self.n_proteins = len(self.protein_names)

        # Making sure the split fractions add up
        if np.abs(np.sum(train_val_test_split_frac) - 1.0) > 1.0E-12:
            train_val_test_split_frac[0] = 1.0 - np.sum(train_val_test_split_frac[1:])

        # Preparing the dataset usable in PyTorch
        self.dataset = self.prepare_torch_dataset(train_val_test_split_frac)

        for _type in ["train", "val", "test"]:
            print(f"Number of datapoints in {_type} dataset = {self.dataset[_type]['target'].size()}")

        # Using data augmentation for the training set
        self.train_set = DatasetAugmentor(torch.utils.data.TensorDataset(self.dataset["train"]["input"],
                                                                         self.dataset["train"]["target"]),
                                          do_brightness_augmentation,
                                          brightness_augmentation_factor)

        # Prepare validation and test sets without data augmentation
        self.val_set = torch.utils.data.TensorDataset(self.dataset["val"]["input"], self.dataset["val"]["target"])
        self.test_set = torch.utils.data.TensorDataset(self.dataset["test"]["input"], self.dataset["test"]["target"])

    @staticmethod
    def load_brightness_data(dataset_file_name, verbose):

        with open(dataset_file_name, "rb") as _file:

            brightness_data = pickle.load(_file)

            _file.close()

        channel_mean = brightness_data.pop("channel_mean", None)
        channel_std = brightness_data.pop("channel_std", None)

        protein_names = list(brightness_data.keys())

        if verbose:
            print("Brightness data loaded for: ")
            print(protein_names)

            for _pr in protein_names:
                print(f"Data for {_pr} with id {brightness_data[_pr]['ID']}")
                print(f"Brightness data size: {brightness_data[_pr]['data'].shape}")
                print(f"Brightness data mean: {np.mean(brightness_data[_pr]['data'], axis=1)}")
                print(f"Brightness data std: {np.std(brightness_data[_pr]['data'], axis=1)}")

            if channel_mean is not None:
                print(f"channel mean = {channel_mean}")
            if channel_std is not None:
                print(f"channel std = {channel_std}")

        return brightness_data, protein_names, channel_mean, channel_std

    @staticmethod
    def append_to_dataset(accum_input, accum_target, data, target, filter_ind):

        filtered_data = torch.from_numpy(data[filter_ind, :].copy()).to(nc.nn_device)
        filtered_target = torch.from_numpy(target[filter_ind].copy()).to(nc.nn_device)

        accum_input = torch.cat((accum_input, filtered_data), dim=0)
        accum_target = torch.cat((accum_target, filtered_target), dim=0)

        return accum_input, accum_target

    def prepare_torch_dataset(self, train_val_test_split_frac) -> object:

        dataset = {}

        for _type in ["train", "val", "test"]:

            dataset[_type] = {"input": torch.empty((0, n_channels), dtype=torch.float32, device=nc.nn_device),
                              "target": torch.empty(0, dtype=torch.long, device=nc.nn_device)}

        split_crit = np.cumsum(train_val_test_split_frac)

        for protein_name in self.protein_names:

            protein_ind = self.brightness_data[protein_name]["ID"]

            data = self.brightness_data[protein_name]["data"].copy().transpose().astype(np.float32)
            target = np.repeat(protein_ind, data.shape[0], axis=0).astype(np.int_)

            random_choice = self.random.random(data.shape[0])

            _filter = {"train": random_choice < split_crit[0],
                       "val": (random_choice >= split_crit[0]) * (random_choice < split_crit[1]),
                       "test": random_choice >= split_crit[1]}

            for _type in ["train", "val", "test"]:
                dataset[_type]["input"], dataset[_type]["target"] = \
                    self.append_to_dataset(dataset[_type]["input"],
                                           dataset[_type]["target"], data, target, _filter[_type])

        return dataset
