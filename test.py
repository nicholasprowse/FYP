import nibabel as nib
import torch
from os.path import join

import util
import numpy as np


def get_train_img(config, i):
    data = np.load(join(config.path, f'train_{i}.npz'))
    return data['data'], data['label']


def main():
    path_img = 'data/Task04_Hippocampus/TrImg_0.nii.gz'
    path_lbl = 'data/Task04_Hippocampus/TrLbl_0.nii.gz'
    data = nib.load(path_img).get_fdata()
    lbl = nib.load(path_lbl).get_fdata()
    print(data.shape)
    print(lbl.shape)

    lbl = np.moveaxis(np.array(util.one_hot(torch.from_numpy(lbl))), 1, 0)
    util.img2gif(data, 2, 'hippocampus.gif')
    util.img2gif(data, 2, 'hippocampus_labelled.gif', label=lbl)


def main2():
    import dataset_preparation
    import preprocessing
    dataset_preparation.prepare_decathlon_dataset('/Users/nicholasprowse/Documents/Engineering/FYP',
                                                  'data', 'Task04_Hippocampus')
    preprocessing.preprocess_dataset('data', 'Task04_Hippocampus', 50*1024**2)


def main4():
    import dataset_preparation
    in_path = '/Volumes/One Touch/orig_med_data'
    out_path = '/Volumes/One Touch/med_data'
    dataset_preparation.prepare_ACDC_dataset(in_path, out_path, 'ACDC')


def main5():
    import preprocessing
    path = 'data'
    preprocessing.preprocess_dataset(path, 'Task04_Hippocampus', 10*1024**2)


if __name__ == '__main__':
    # util.generate_plot([300, 301, 302, 303, 304, 305, 306, 307], 'beta')
    main5()


# Start shape: (512, 512, 49)
# Spacing: (0.917969, 0.917969, 5.0)
# Target spacing: [0.7988280057907104, 0.7988280057907104, 1.5]
# Final shape: (1, 588, 588, 163)
# Processed 0/303 training images
# Start shape: (512, 512, 66)
# Spacing: (0.951172, 0.951172, 3.75)
# Target spacing: [0.7988280057907104, 0.7988280057907104, 1.5]
# Final shape: (1, 610, 610, 165)
