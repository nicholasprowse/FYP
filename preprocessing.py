import shutil

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
            label = np.swapaxes(label, dim-1, 0)[min_idx:max_idx + 1]

    # All the swapped dimensions result in all dimensions still being in order, except the final dimension is
    # now the first dimension, so we move that back to the end
    data = np.moveaxis(data, 0, len(data.shape) - 1)
    if label is not None:
        label = np.moveaxis(label, 0, len(data.shape) - 2)
    return data, label


def prepare_dataset(path, name):
    """
    Crops all images in the given dataset to the non-zero region, then saves all of the dataset into a single
    .npz file. Data set must be organised as follows. All files in one folder, training data has name trImg_#.nii.gz
    where # is a decimal number, training labels have name trLbl_#.nii.gz, with corresponding number to its image.
    Test image has name tsImg_#.nii.gz
    :param path:
    :param name:
    :return:
    """
    config = first_pass(path, name)
    second_pass(config)


def first_pass(path, name):
    """
    In the first pass of the data preparation process, the images are cropped to the non zero region, sizes and image
    spacings are determined and the number of classes is determined
    """
    config = ml_collections.ConfigDict()
    dataset_path = join(path, name)
    processed_path = join(path, f'{name}_processed')

    config.path = processed_path

    if not os.path.exists(processed_path):
        os.mkdir(processed_path)

    shutil.copy(join(dataset_path, 'data.json'), join(processed_path, 'data.json'))
    dataset_json = json.load(open(join(dataset_path, 'data.json')))
    shapes = np.zeros((dataset_json['num_train'], 3))
    train_spacing = np.zeros((dataset_json['num_train'], 3))
    classes = np.array([])
    for i in range(dataset_json['num_train']):
        nib_data = nib.load(join(dataset_path, f'trImg_{i}.nii.gz'))
        nib_label = nib.load(join(dataset_path, f'trLbl_{i}.nii.gz'))
        img = nib_data.get_fdata()
        lbl = nib_label.get_fdata()
        # Make sure data is 4 dimensional, with channels as first dimension
        if img.ndim == 3:
            img = np.expand_dims(img, 0)
        else:
            img = np.moveaxis(img, 3, 0)

        train_spacing[i, :] = np.array(nib_data.header.get_zooms()[0:3])
        img, lbl = crop(img, lbl)
        classes = np.union1d(classes, lbl)
        shapes[i, :] = np.array(img.shape[1:4])

        # Compressed npz files are around twice as slow (to read) as npy files
        # Test results: npy: 0.213s, npz: 0.311s, compressed npz: 0.727s
        data_path = join(processed_path, f'train_{i}.npz')
        if not os.path.exists(data_path):  # To save time, only save the data if not already there
            np.savez_compressed(data_path, data=img, label=lbl)

    config.train_spacings = train_spacing
    config.median_spacing = np.median(train_spacing, axis=0)
    config.median_shape = np.median(shapes, axis=0)
    config.isotropy = np.max(config.median_spacing) / np.min(config.median_spacing)
    config.anisotropic_axis = np.argmax(config.median_spacing)
    config.tenth_percentile_spacing = np.percentile(train_spacing[config.anisotropic_axis, :], 10)
    # config.dims
    config.classes = list(classes)
    config.ct = dataset_json['ct']
    config.num_train = dataset_json['num_train']

    config.target_spacing = config.median_spacing
    if config.isotropy >= 3:
        config.target_spacing[config.anisotropic_axis] = config.tenth_percentile_spacing
    test_spacing = np.zeros((dataset_json['num_test'], 3))
    for i in range(dataset_json['num_test']):
        nib_img = nib.load(join(dataset_path, f'tsImg_{i}.nii.gz'))
        img = nib_img.get_fdata()
        # Make sure data is 4 dimensional, with channels as first dimension
        if img.ndim == 3:
            img = np.expand_dims(img, 0)
        else:
            img = np.swapaxes(img, 0, 3)

        test_spacing[i, :] = np.array(nib_img.header.get_zooms()[0:3])
        img, _ = crop(img, None)
        data_path = join(processed_path, f'test_{i}.npz')
        if not os.path.exists(data_path):
            np.savez_compressed(data_path, data=img)

    config.test_spacings = test_spacing
    return config


def second_pass(config):
    """
    In the second pass of the data preparation process, we convert the labels into a one hot encoding, resample
    the images to have a consistent pixel spacing and normalise the intensity of the images
    """
    dataset_json = json.load(open(join(config.path, 'data.json')))
    for i in range(dataset_json['num_train']):
        data_path = join(config.path, f'train_{i}.npz')
        with np.load(data_path) as data:
            img = data['data']
            lbl = data['label']
            img = normalise(img, config)
            img, lbl = resize(img, config.train_spacings[i, :], config, label=lbl)
            lbl = util.one_hot(lbl, classes=config.classes)
            np.savez_compressed(data_path, data=img, label=lbl)

    for i in range(dataset_json['num_test']):
        data_path = join(config.path, f'test_{i}.npz')
        with np.load(data_path) as data:
            img = data['data']

            img = normalise(img, config)
            img, _ = resize(img, config.test_spacings[i, :], config)
            np.savez_compressed(data_path, data=img)


def resize(data, spacing, config, label=None):
    new_size = data.shape[1:] * spacing / config.target_spacing

    if np.all(new_size == data.shape[1:]):
        return data, label

    new_data = np.zeros([data.shape[0]] + new_size)
    new_label = None
    if label is not None:
        new_label = np.uint8(np.zeros([label.shape[0]] + new_size))
    if config.isotropy >= 3:
        # z-axis is nearest neighbor
        for c in range(len(data.shape[0])):
            # Resize with order 3 along just the first 2 dimensions
            first_size = np.array([new_size[0], new_size[1], data.shape[2]])
            first_data = skimage.transform.resize(data[c], first_size, order=3)
            new_data[c] = skimage.transform.resize(first_data, new_size, order=0)

            if label is not None:
                scaled_label = skimage.transform.resize(np.float(new_label[c]), first_size, order=3)
                # Resize with nearest neighbor along 3rd dimension
                scaled_label = skimage.transform.resize(scaled_label, new_size, order=0)
                new_label[c] = np.uint8(np.around(scaled_label))

    else:
        for c in range(len(data.shape[0])):
            new_data[c] = skimage.transform.resize(data[c], new_size, order=3)
            if label is not None:
                scaled_label = skimage.transform.resize(np.float(new_label[c]), new_size, order=3)
                new_label[c] = np.uint8(np.around(scaled_label))

    return new_data, new_label


def normalise(data, config):
    """
    Channels that are of the CT modality type are clipped within the 0.5 and 99.5 percentiles and all channels are
    normalised to unit stdev and zero mean
    :param data: numpy array with channels as the first dimension
    :param config: config dictionary, needed for the modalities of each channel
    :return: normalised data
    """
    for i in range(data.shape[0]):
        if config.ct[i]:
            percentile_99_5 = np.percentile(data[i], 99.5)
            percentile_0_5 = np.percentile(data[i], 0.5)
            data[i] = np.clip(data[i], percentile_0_5, percentile_99_5)
        mean = np.mean(data[i])
        std = np.std(data[i])
        data[i] = (data[i] - mean) / std

    return data
