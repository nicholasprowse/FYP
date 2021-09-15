import nibabel as nib
import torch
from matplotlib import pyplot as plt
import json
from os.path import join
import os

import model
import util
import numpy as np
import ml_collections
import dataset_preparation
import skimage.transform
import preprocessing
from model import VisionTransformer, get_r50_b16_config


def get_train_img(config, i):
    data = np.load(join(config.path, f'train_{i}.npz'))
    return data['data'], data['label']


def main():
    path_img = '/Volumes/One Touch/orig_med_data/ACDC/training/patient093/patient093_frame14.nii.gz'
    path_lbl = '/Volumes/One Touch/orig_med_data/ACDC/training/patient093/patient093_frame14_gt.nii.gz'
    data = nib.load(path_img).get_fdata()
    lbl = nib.load(path_lbl).get_fdata()
    print(data.shape)
    print(lbl.shape)


    # lbl = np.moveaxis(np.array(util.one_hot(torch.from_numpy(lbl))), 1, 0)
    util.img2gif(data, 2, 'heart.gif')
    util.img2gif(lbl, 2, 'heart_label.gif')


def main2():
    import preprocessing
    path = '/Users/nicholasprowse/Documents/Engineering/FYP/data'
    preprocessing.prepare_dataset(path, 'Task04_Hippocampus', 4*1024**3)


def main4():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    config = get_r50_b16_config(dims=3, img_size=[64, 64, 64])
    config.skip_channels[-1] = 1
    config.input_channels = 1
    vt = VisionTransformer(config)
    batch = torch.zeros([2, 1, 59, 67, 41])
    print(vt(batch).shape)


if __name__ == '__main__':
    main2()


def load_dataset_fingerprint(path):
    config = ml_collections.ConfigDict()
    dataset = json.load(open(join(path, 'dataset.json')))
    modalities = [i.lower() for i in dataset['modality']]
    config.percentile_clip = 'ct' in modalities
    isotropic = True

