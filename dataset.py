from torch.utils.data import dataset
import json
from os.path import join
import torch
import numpy as np



class Loader3D(dataset):
    def __init__(self, path, transform=None):
        self.json = json.load(open(join(path, 'data.json')))
        self.len = self.json['num_train']
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        with np.load(join(self.config.path, f'train_{i}.npz')) as data:
            img = torch.from_numpy(data['img'])
            label = torch.from_numpy(data['label'])
        img, label = self.transform((img, label))
        return img, label



class Loader2D(dataset):
    pass

