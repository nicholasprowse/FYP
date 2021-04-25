import nibabel as nib
import torch
from matplotlib import pyplot as plt
import json
import os
import util


def main():
    path = '/Volumes/One Touch/Medical Decathlon Data/Task01_BrainTumour'
    dataset = json.load(open(os.path.join(path, 'dataset.json')))
    img = nib.load(os.path.join(path, dataset['training'][1]['image'])).get_fdata()
    label = nib.load(os.path.join(path, dataset['training'][1]['label'])).get_fdata()

    img = torch.Tensor(img)
    label = torch.Tensor(label)
    # print(label[0].shape)
    channel = 2
    encoding = util.one_hot(label, 4)
    util.img2gif(img[:, :, :, channel], 2, 'tumor.gif', label=encoding)
    util.img2gif(img[:, :, :, channel], 2, 'brain.gif')
    # img2gif(label, 2, 'label.gif')

    for i in range(4):
        plt.imshow(encoding[i, :, :, 75], cmap='gray')
        plt.savefig(f'encoding{i}.png')


if __name__ == '__main__':
    main()
