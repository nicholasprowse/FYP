from torch.utils.data import Dataset
import json
from os.path import join
import torch
import numpy as np
from functools import reduce
from util import center_crop
from torch.nn.functional import pad


class Loader3D(Dataset):
    def __init__(self, path, transform=None):
        self.json = json.load(open(join(path, 'data.json')))
        self.len = self.json['num_train']
        self.img_size = self.json['shape']
        self.patch_size = np.array(self.json['patch_size'])
        self.path = path
        # This is the number of patches along each dimension
        self.patch_grid_shape = torch.tensor(self.json['patch_grid_shape'])
        self.num_patches = int(reduce(lambda a, b: a*b, self.patch_grid_shape))
        self.transform = transform

    def __len__(self):
        return self.len * self.num_patches

    def __getitem__(self, i):
        image_id = i // self.num_patches
        with np.load(join(self.path, f'train_{image_id}.npz')) as data:
            img = data['data']
            lbl = data['label']

        # pad tensor to be img_size
        img = center_crop(img, self.img_size)
        lbl = center_crop(lbl, self.img_size[1:])

        # The patch coordinates of the patch to extract
        indexes = np.array([0, 0, 0])
        products = [1, 1, 1, 1]
        for j in range(3):
            products[j+1] = products[j] * int(self.patch_grid_shape[j])
            indexes[j] = (i % products[j+1]) // products[j]

        lb = (indexes*self.patch_size).astype(int)
        ub = ((indexes+1)*self.patch_size).astype(int)
        img = img[:, lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]]
        lbl = lbl[lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]]

        if self.transform is not None:
            img, label = self.transform((img, lbl))

        return torch.from_numpy(img).float(), torch.from_numpy(lbl).long()


class Loader2D(Dataset):
    def __init__(self, path, transform=None):
        self.json = json.load(open(join(path, 'data.json')))
        self.img_size = self.json['shape']
        self.depths = self.json['depths']
        self.len = self.json['total_depth']
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
            img = torch.from_numpy(data['data'])[:, :, :, i]
            lbl = torch.from_numpy(data['label'])[:, :, i]

        img = center_crop(img, self.img_size[:3])
        lbl = center_crop(lbl, self.img_size[1:3])
        return img.float(), lbl.long()
