from torch.utils.data import Dataset
import json
from os.path import join
import torch
import numpy as np
from functools import reduce
import util


class Loader3D(Dataset):
    def __init__(self, path, transform=None):
        self.data_config = json.load(open(join(path, 'data.json')))
        self.len = self.data_config['num_train']
        self.img_size = np.array(self.data_config['shape'])
        self.patch_size = np.array(self.data_config['patch_size'])

        self.patches_along_each_axis = self.data_config['patches_along_each_axis']
        self.patch_overlap = self.data_config['patch_overlap']

        self.channels = self.data_config['channels']
        self.path = path
        # This is the number of patches along each dimension
        self.num_patches = int(reduce(lambda a, b: a * b, self.patches_along_each_axis))
        self.transform = transform
        self.do_transform = transform is not None

    def __len__(self):
        return self.len * self.num_patches

    def train(self):
        self.do_transform = self.transform is not None

    def eval(self):
        self.do_transform = False

    def __getitem__(self, i):
        image_id = i // self.num_patches
        with np.load(join(self.path, f'train_{image_id}.npz')) as data:
            image = data['image']
            label = data['label']

        # pad tensor to be img_size
        image = util.center_crop(image, [self.channels] + list(self.img_size))
        label = util.center_crop(label, self.img_size)
        # The patch coordinates of the patch to extract
        indexes = index_to_patch_location(i, self.patches_along_each_axis)

        lb = (np.minimum(indexes * self.patch_size - indexes * self.patches_along_each_axis,
                         self.img_size - self.patch_size)).astype(int)
        ub = (lb + self.patch_size).astype(int)
        image = image[:, lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]]
        label = label[lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]]

        if self.do_transform:
            image, label = self.transform((image, label))

        # We must copy the numpy arrays as torch does not support numpy arrays with negative strides (flipped arrays)
        return torch.from_numpy(image.copy()).float(), torch.from_numpy(label.copy()).long()


class Loader2D(Dataset):
    def __init__(self, path, transform=None):
        self.data_config = json.load(open(join(path, 'data.json')))
        self.img_size = self.data_config['shape']
        self.depths = self.data_config['depths']
        self.channels = self.data_config['channels']
        self.len = int(sum(self.depths))
        self.path = path
        self.transform = transform
        self.do_transform = transform is not None

    def __len__(self):
        return self.len

    def train(self):
        self.do_transform = self.transform is not None

    def eval(self):
        self.do_transform = False

    def __getitem__(self, i):
        image_id = 0
        for j, d in enumerate(self.depths):
            if i >= d:
                i -= int(d)
            else:
                image_id = j
                break

        with np.load(join(self.path, f'train_{image_id}.npz')) as data:
            image = data['image'][:, :, :, i]
            label = data['label'][:, :, i]

        image = util.center_crop(image, [self.channels] + self.img_size[:2])
        label = util.center_crop(label, self.img_size[:2])

        if self.do_transform:
            image, label = self.transform((image, label))

        # We must copy the numpy arrays as torch does not support numpy arrays with negative strides (flipped arrays)
        return torch.from_numpy(image.copy()).float(), torch.from_numpy(label.copy()).long()


def index_to_patch_location(index, patches_along_each_axis):
    indexes = np.array([0, 0, 0])
    products = [1, 1, 1, 1]
    for j in range(3):
        products[j + 1] = products[j] * int(patches_along_each_axis[j])
        indexes[j] = (index % products[j + 1]) // products[j]
    return indexes
