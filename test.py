import nibabel as nib
import torch
from matplotlib import pyplot as plt
import json
from os.path import join
import os
import util
import numpy as np
import ml_collections
import dataset
import skimage.transform
import preprocessing
from model import VisionTransformer, get_r50_b16_config


def get_train_img(config, i):
    data = np.load(join(config.path, f'train_{i}.npz'))
    return data['data'], data['label']


def main():
    import resnetV2
    print(resnetV2.ResNetV2(block_units=(3, 4, 9), width_factor=1))
    quit()
    path_img = '/Volumes/One Touch/Medical Decathlon Data/Task01_BrainTumour/imagesTr/BRATS_004.nii.gz'
    # path_lbl = '/Volumes/One Touch/Medical Decathlon Data/Task01_BrainTumour/labelsTr/BRATS_004.nii.gz'
    data = nib.load(path_img).get_fdata()
    lbl = nib.load('/Volumes/One Touch/Medical Decathlon Data/Task01_BrainTumour/labelsTr/BRATS_004.nii.gz').get_fdata()
    data = np.moveaxis(data, 3, 0)
    lbl = util.one_hot(lbl)
    data, lbl = preprocessing.crop(data, label=lbl)
    util.img2gif(data[2], 2, 'brain.gif')
    util.img2gif(data[2], 2, 'brain_labelled.gif', label=lbl)
    # label = nib.load(path_lbl)
    # data = np.swapaxes(data, 0, 3)
    # cropped, _ = preprocessing.crop(data)
    # util.img2gif(data[2], 1, 'brain.gif')
    # util.img2gif(cropped[2], 1, 'brain_cropped.gif')
    # util.img2gif(data.get_fdata()[:, :, :, 0], 2, 'brain_tumour0.gif', util.one_hot(label.get_fdata()))
    # util.img2gif(data.get_fdata()[:, :, :, 0], 2, 'brain0.gif')
    # util.img2gif(data.get_fdata()[:, :, :, 1], 2, 'brain_tumour1.gif', util.one_hot(label.get_fdata()))
    # util.img2gif(data.get_fdata()[:, :, :, 1], 2, 'brain1.gif')
    # util.img2gif(data.get_fdata()[:, :, :, 2], 2, 'brain_tumour2.gif', util.one_hot(label.get_fdata()))
    # util.img2gif(data.get_fdata()[:, :, :, 2], 2, 'brain2.gif')
    # util.img2gif(data.get_fdata()[:, :, :, 3], 2, 'brain_tumour3.gif', util.one_hot(label.get_fdata()))
    # util.img2gif(data.get_fdata()[:, :, :, 3], 2, 'brain3.gif')
    # label = nib.load('/Volumes/One Touch/ACDC/training/patient001/patient001_frame01_gt.nii.gz')
    # print(data.get_fdata().shape)
    # util.img2gif(data.get_fdata(), 2, 'heart.gif', label=util.one_hot(label.get_fdata(), num_classes=4))
    # name = 'Task04_Hippocampus'
    # dataset.prepare_synapse_dataset('/Volumes/One Touch', 'synapse')


def main2():
    img_size = [126, 96, 192]
    img = torch.zeros([1 if i < 2 else img_size[i-2] for i in range(5)])
    print("Image Created")
    net = VisionTransformer(get_r50_b16_config(dims=3, img_size=img_size))
    print("Network Created")
    out = net(img)
    print(out.shape)
    pass


if __name__ == '__main__':
    main2()


def load_dataset_fingerprint(path):
    config = ml_collections.ConfigDict()
    dataset = json.load(open(join(path, 'dataset.json')))
    modalities = [i.lower() for i in dataset['modality']]
    config.percentile_clip = 'ct' in modalities
    isotropic = True

