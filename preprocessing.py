import ml_collections
from os.path import join
import os
import json
import numpy as np
import nibabel as nib
import util
import skimage.transform


def crop(data, label=None):
    """
    Crops the given image to the smallest region containing all non-zero data. First dimension is ignored, as this
    is assumed to be the channels.
    :param data: Numpy array of any dimensionality
    :param label: optional argument that will get cropped to the same dimensions as the data
    :return: Cropped version of input data with only the nonzero region
    """
    # Get indices of all nonzero elements
    idx = np.nonzero(data)

    for dim in range(1, len(data.shape)):
        # For each dimension, compute the minimum and maximum nonzero index in that dimension
        min_idx = np.min(idx[dim])
        max_idx = np.max(idx[dim])
        # Swap given dimension to be first dimension, and slice out the nonzero region
        data = np.swapaxes(data, dim, 0)[min_idx:max_idx + 1]
        if label is not None:
            label = np.swapaxes(label, dim, 0)[min_idx:max_idx + 1]

    # All the swapped dimensions result in all dimensions still being in order, except the final dimension is
    # now the first dimension, so we move that back to the end
    data = np.moveaxis(data, 0, len(data.shape) - 1)
    if label is not None:
        label = np.moveaxis(label, 0, len(data.shape) - 1)
    return data, label


def prepare_dataset(path, dataset):
    """
    Crops all images in the given dataset to the non-zero region, then saves all of the dataset into a single
    .npz file. Data set must be organised as follows. All files in one folder, training data has name trImg#.nii.gz
    where # is a decimal number, training labels have name trLbl#.nii.gz, with corresponding number to its image.
    Test image has name tsImg#.nii.gz
    :param path:
    :param dataset:
    :return:
    """
    config = ml_collections.ConfigDict()
    dataset_path = join(path, dataset)
    cropped_path = join(path, f'{dataset}_processed')

    config.path = cropped_path

    if not os.path.exists(cropped_path):
        os.mkdir(cropped_path)

    dataset_json = json.load(open(join(dataset_path, 'dataset.json')))
    shapes = np.zeros((dataset_json['numTraining'], 3))
    spacing = np.zeros((dataset_json['numTraining'], 3))
    for i in range(dataset_json['numTraining']):
        img_name = dataset_json['training'][i]['image']
        lbl_name = dataset_json['training'][i]['label']

        nib_data = nib.load(join(dataset_path, img_name))
        nib_label = nib.load(join(dataset_path, lbl_name))
        img = nib_data.get_fdata()
        lbl = nib_label.get_fdata()

        # Make sure data is 4 dimensional, with channels as first dimension
        if img.ndim == 3:
            np.expand_dims(img, 0)
        else:
            np.moveaxis(img, 3, 0)

        spacing[i, :] = np.array(nib_data.header.get_zooms()[0:3])
        img, lbl = crop(img, lbl)

        lbl = util.one_hot(lbl)
        shapes[i, :] = np.array(img.shape[1:4])

        # Compressed npz files are around twice as slow (to read) as npy files
        # Test results: npy: 0.213s, npz: 0.311s, compressed npz: 0.727z
        data_path = join(cropped_path, f'train_{i}.npz')
        if not os.path.exists(data_path):      # To save time, only save the data if not already there
            np.savez_compressed(data_path, data=img, label=lbl)

    config.train_spacings = spacing
    config.median_spacing = np.median(spacing, axis=0)
    config.median_shape = np.median(shapes, axis=0)
    config.isotropy = np.max(config.median_spacing) / np.min(config.median_spacing)
    config.anisotropic_axis = np.argmax(config.median_spacing)
    config.tenth_percentile_spacing = np.percentile(spacing[config.anisotropic_axis, :], 10)

    config.target_spacing = config.median_spacing
    if config.isotropy >= 3:
        config.target_spacing[config.anisotropic_axis] = config.tenth_percentile_spacing

    print(config.median_spacing)
    print(config.median_shape)
    print(config.isotropy)
    print(config.anisotropic_axis)
    print(config.tenth_percentile_spacing)
    print(config.target_spacing)

    for i in range(dataset_json['numTest']):
        img = nib.load(join(dataset_path, dataset_json['test'][i])).get_fdata()
        # Make sure data is 4 dimensional, with channels as first dimension
        if img.ndim == 3:
            np.expand_dims(img, 0)
        else:
            np.swapaxes(img, 0, 3)

        img, _ = crop(img, None)
        data_path = join(cropped_path, f'test_{i}.npz')
        if not os.path.exists(data_path):
            np.savez_compressed(data_path, img)

    return config


def resize(data, label, spacing, config):
    new_size = data.shape * spacing / config.target_spacing
    if np.all(new_size == data.shape):
        return data, label

    new_data = np.zeros([data.shape[0]] + new_size)
    new_label = np.uint8(np.zeros([label.shape[0]] + new_size))
    if config.isotropy >= 3:
        # z-axis is nearest neighbor
        for c in range(len(data.shape[0])):
            # Resize with order 3 along just the first 2 dimensions
            first_size = np.array([new_size[0], new_size[1], data.shape[2]])
            first_data = skimage.transform.resize(data[c], first_size, order=3)
            scaled_label = skimage.transform.resize(np.float(new_label[c]), first_size, order=3)
            # Resize with nearest neighbor along 3rd dimension
            scaled_label = skimage.transform.resize(scaled_label, new_size, order=0)
            new_label[c] = np.uint8(np.around(scaled_label))
            new_data[c] = skimage.transform.resize(first_data, new_size, order=0)
        pass
    else:
        for c in range(len(data.shape[0])):
            new_data[c] = skimage.transform.resize(data[c], new_size, order=3)
            scaled_label = skimage.transform.resize(np.float(new_label[c]), new_size, order=3)
            new_label[c] = np.uint8(np.around(scaled_label))
