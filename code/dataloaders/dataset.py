import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from skimage import transform as sk_trans
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image
from dataloaders.prior_mix import prior_mix


class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        labeled_idxs=None,
        transform = None
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.labeled_idxs = labeled_idxs
        self.transform = transform
        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "")
                                for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "")
                                for item in self.sample_list]

        print("dataset total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/data/slices/{}.h5".format(case), "r")
            image = h5f["image"][:]
            label = h5f["label"][:]
            mix_idx = random.choice(self.labeled_idxs)
            # print(f"id: {idx}  MIX_id{mix_idx}")
            mix_case = self.sample_list[mix_idx]
            mix_h5f = h5py.File(self._base_dir +
                            "/data/slices/{}.h5".format(mix_case), "r")
            mix_image = mix_h5f["image"][:]
            mix_label = mix_h5f["label"][:]
            sample = self.transform(image, label, mix_image, mix_label)
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
            image = h5f["image"][:]
            label = h5f["label"][:]
            sample = {"image": image, "label": label}
        

            
        sample["idx"] = idx
        

        return sample





def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label





class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
