from torch.utils.data import Dataset
import json
from os.path import join
import torch
import numpy as np
from functools import reduce


class Loader3D(Dataset):
    def __init__(self, path, transform=None):
        self.json = json.load(open(join(path, 'data.json')))
        self.len = self.json['num_train']
        self.path = path
        # This is the number of patches along each dimension
        self.patch_grid_shape = torch.tensor(self.json['patch_grid_shape'])
        self.num_patches = reduce(lambda a, b: a*b, self.patch_grid_shape)
        self.transform = transform

    def __len__(self):
        return self.len * self.num_patches

    def __getitem__(self, i):
        image_id = i // self.num_patches
        with np.load(join(self.path, f'train_{image_id}.npz')) as data:
            img = torch.from_numpy(data['data'])
            label = torch.from_numpy(data['label'])
        # The patch coordinates of the patch to extract
        indexes = torch.tensor([0, 0, 0])
        products = [1, 1, 1, 1]
        for j in range(3):
            products[j+1] = products[j] * int(self.patch_grid_shape[j])
            indexes[j] = (i % products[j+1]) // products[j]
        patch_size = torch.tensor(img.shape[1:4]) / self.patch_grid_shape

        lb = torch.round(indexes*patch_size).to(dtype=torch.uint8)
        ub = torch.round((indexes+1)*patch_size).to(dtype=torch.uint8)
        img = img[:, lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]]
        label = label[:, lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]]

        if self.transform is not None:
            img, label = self.transform((img, label))

        return img, label


class Loader2D(Dataset):
    def __init__(self, path, transform=None):
        self.json = json.load(open(join(path, 'data.json')))
        self.depths = self.json['depths']
        self.len = self.json['total_layers']
        self.path = path
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        image_id = 0
        for j, d in enumerate(self.depths):
            if i >= d:
                i -= d
            else:
                image_id = j
                break

        with np.load(join(self.path, f'train_{image_id}.npz')) as data:
            img = torch.from_numpy(data['data'])
            label = torch.from_numpy(data['label'])

        return img[:, :, :, i], label[:, :, :, i]