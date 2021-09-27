from os.path import join
import os
import json

import numpy as np
import skimage.transform
from functools import reduce

import util


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

    # difference in the number of dims in label and data, which is used to make sure the same dimensions are affected
    # in both arrays

    offset = data.ndim - label.ndim if label is not None else 0
    for dim in range(1, data.ndim):
        # For each dimension, compute the minimum and maximum nonzero index in that dimension
        min_idx = np.min(idx[dim])
        max_idx = np.max(idx[dim])
        # Swap given dimension to be first dimension, and slice out the nonzero region
        data = np.swapaxes(data, dim, 0)[min_idx:max_idx + 1]
        if label is not None:
            label = np.swapaxes(label, dim - offset, 0)[min_idx:max_idx + 1]

    # All the swapped dimensions result in all dimensions still being in order, except the final dimension is
    # now the first dimension, so we move that back to the end
    data = np.moveaxis(data, 0, -1)
    if label is not None:
        label = np.moveaxis(label, 0, data.ndim - offset - 1)
    return data, label


def get_crops(image):
    """
    Returns the amount cropped on each dimension when the image is cropped to its non-zero size. This is formatted so
    that you can use it directly in torch.pad to undo the crop
    """
    idx = np.nonzero(image)
    total_crop = []
    for dim in range(1, image.ndim):
        # For each dimension, compute the minimum and maximum nonzero index in that dimension
        min_idx = np.min(idx[dim])
        max_idx = np.max(idx[dim])
        padding = [min_idx, image.shape[dim] - max_idx - 1]
        total_crop = padding + total_crop

    total_crop = [int(i) for i in total_crop]
    return total_crop


def get_nonzero_size(image):
    """
    Finds the size of the image after it has been cropped to its nonzero region without actually performing the crop
    :param image: Numpy array of any dimensionality
    :return: Shape of the nonzero region in the image as a tuple
    """
    # Get indices of all nonzero elements
    idx = np.nonzero(image)
    shape = [0] * (image.ndim - 1)
    # Find the max and min nonzero element in each dim and set the shape accordingly
    for dim in range(1, image.ndim):
        shape[dim-1] = np.max(idx[dim]) - np.min(idx[dim]) + 1
    return tuple(shape)


def prepare_dataset(path, name, memory_constraint):
    """
    Crops all images in the given dataset to the non-zero region, then saves all of the dataset into a single
    .npz file. Data set must be organised as follows. All files in one folder, training data has name trImg_#.nii.gz
    where # is a decimal number, training labels have name trLbl_#.nii.gz, with corresponding number to its image.
    Test image has name tsImg_#.nii.gz
    :param path:
    :param name:
    :param memory_constraint:
    :return:
    """
    config = first_pass(path, name)
    second_pass(config, memory_constraint)


def first_pass(path, name):
    """
    In the first pass of the data preparation process, the images are cropped to the non zero region, sizes and image
    spacings are determined and the number of classes is determined
    """
    import nibabel as nib
    dataset_path = join(path, name)
    processed_path = join(path, f'{name}_processed')
    config = json.load(open(join(dataset_path, 'data.json')))
    config['path'] = processed_path
    config['raw_path'] = path

    if not os.path.exists(processed_path):
        os.mkdir(processed_path)

    shapes = np.zeros((config['num_train'], 4))
    train_spacing = np.zeros((config['num_train'], 3))
    classes = np.array([])
    for i in range(config['num_train']):
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
        shapes[i, :] = np.array(img.shape)

        # Compressed npz files are around twice as slow (to read) as npy files
        # Test results: npy: 0.213s, npz: 0.311s, compressed npz: 0.727s
        data_path = join(processed_path, f'train_{i}.npz')
        np.savez_compressed(data_path, data=img, label=lbl)

    config['train_spacings'] = train_spacing
    config['median_spacing'] = np.median(train_spacing, axis=0)
    config['shape'] = np.max(shapes, axis=0)
    config['isotropy'] = np.max(config['median_spacing']) / np.min(config['median_spacing'])
    # We artificially inflate the third axes spacing so that the anisotropic defaults to the third axis if all other
    # # axes are equal
    config['median_spacing'][2] *= 1.01
    config['anisotropic_axis'] = int(np.argmax(config['median_spacing']))
    config['median_spacing'][2] /= 1.01
    config['tenth_percentile_spacing'] = np.percentile(train_spacing[config['anisotropic_axis'], :], 10)
    config['n_classes'] = len(list(classes))

    config['target_spacing'] = config['median_spacing']
    if config['isotropy'] >= 3:
        config['target_spacing'][config['anisotropic_axis']] = config['tenth_percentile_spacing']

    test_spacing = np.zeros((config['num_test'], 3))
    for i in range(config['num_test']):
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
        np.savez_compressed(data_path, data=img)

    config['test_spacings'] = test_spacing
    return config


def volume(shape):
    """returns the total number of voxels in an image"""
    return reduce(lambda a, b: a * b, shape)


def second_pass(config, memory_constraint):
    """
    In the second pass of the data preparation process, we convert the labels into a one hot encoding, resample
    the images to have a consistent pixel spacing and normalise the intensity of the images. We also determine the patch
    size based on maximum GPU memory allocation for the batch
    """
    config['depths'] = [0] * config['num_train']
    config['total_depth'] = 0
    for i in range(config['num_train']):
        data_path = join(config['path'], f'train_{i}.npz')
        with np.load(data_path) as data:
            img = data['data']
            lbl = data['label']
            img = normalise(img, config)

            # Move anisotropic axis to the third spatial axis
            img = np.swapaxes(img, 3, config['anisotropic_axis'] + 1)
            lbl = np.swapaxes(lbl, 2, config['anisotropic_axis'])

            img, lbl = resize(img, config['train_spacings'][i, :], config, label=lbl)

            config['depths'][i] = img.shape[-1]
            config['total_depth'] += img.shape[-1]

            np.savez_compressed(data_path, data=img, label=lbl)

    for i in range(config['num_test']):
        data_path = join(config['path'], f'test_{i}.npz')
        with np.load(data_path) as data:
            img = data['data']

            img = normalise(img, config)
            img, _ = resize(img, config['test_spacings'][i, :], config)
            np.savez_compressed(data_path, data=img)

    # swap values in shape and spacings, so they are still correct after the anisotropic axis has been changed
    config['median_spacing'][2], config['median_spacing'][config['anisotropic_axis']] \
        = config['median_spacing'][config['anisotropic_axis']], config['median_spacing'][2]
    config['shape'][3], config['shape'][config['anisotropic_axis']+1] \
        = config['shape'][config['anisotropic_axis']+1], config['shape'][3]
    # Limits on the batch size
    min_batch_size = 2
    max_dataset_coverage_of_batch = 0.05
    max_voxels_in_batch = max_dataset_coverage_of_batch * config['num_train'] * volume(config['shape'])
    channels = config['shape'][0]
    patch_grid_shape = np.array([1, 1, 1])
    patch_size = np.ceil(config['shape'][1:4] / patch_grid_shape)
    while min_batch_size * 4 * channels * volume(patch_size) >= memory_constraint:
        largest_dim = np.argmax(patch_size)
        patch_grid_shape[largest_dim] *= 2
        patch_size = np.ceil(config['shape'][1:4] / patch_grid_shape)

    # determine the batch size for both 2D and 3D networks
    config['shape'][1:] = patch_size * patch_grid_shape
    config['patch_size'] = patch_size
    config['patch_grid_shape'] = patch_grid_shape
    batch_size_from_memory = np.floor(memory_constraint / (4 * channels * volume(patch_size)))
    batch_size_from_coverage = max_voxels_in_batch / (channels * volume(config['shape'][1:4] / patch_grid_shape))
    config['batch_size3D'] = int(min(batch_size_from_memory, batch_size_from_coverage))

    slice_size = config['shape'][1] * config['shape'][2]
    batch_size_from_memory = np.floor(memory_constraint / (4 * channels * slice_size))
    batch_size_from_coverage = max_voxels_in_batch / (channels * slice_size)
    config['batch_size2D'] = int(min(batch_size_from_memory, batch_size_from_coverage))

    del config['train_spacings']
    del config['test_spacings']
    # Convert numpy types to python types for JSON serialization
    for key, value in config.items():
        if type(value) == np.ndarray:
            config[key] = value.tolist()
        if type(value) == np.int64:
            config[key] = int(value)
        if type(value) == np.float64:
            config[key] = float(value)

    with open(join(config['path'], 'data.json'), 'w') as outfile:
        json.dump(config, outfile)


def obtain_dataset_fingerprint(path, name, memory_constraint):
    import nibabel as nib
    dataset_path = join(path, name)
    processed_path = join(path, f'{name}_processed')
    config = json.load(open(join(dataset_path, 'data.json')))
    config['path'] = processed_path
    config['raw_path'] = join(path, name)

    if not os.path.exists(processed_path):
        os.mkdir(processed_path)

    train_spacing = np.zeros((config['num_train'], 3))
    shapes = np.zeros((config['num_train'], 3))
    class_frequency = np.zeros(config['n_classes'])

    # First pass through the data is just to obtain the spacings of all the images
    for i in range(config['num_train']):
        image_nifti = nib.load(join(dataset_path, f'trImg_{i}.nii.gz'))
        image = image_nifti.get_fdata()
        train_spacing[i, :] = np.array(image_nifti.header.get_zooms()[0:3])
        image = np.expand_dims(image, 0) if image.ndim == 3 else np.moveaxis(image, -1, -0)
        shapes[i, :] = get_nonzero_size(image)
        config['channels'] = image.shape[0]

        label = nib.load(join(dataset_path, f'trLbl_{i}.nii.gz')).get_fdata()
        unique, counts = np.unique(label, return_counts=True)
        for j, clazz in enumerate(unique):
            class_frequency[int(clazz)] += counts[j] / volume(label.shape)

    class_frequency /= config['num_train']
    config['class_weights'] = 1 / ((class_frequency ** 0.3) * (np.sum(class_frequency ** -0.3)))
    # Calculate various dataset fingerprints based on this information
    median_spacing = np.median(train_spacing, axis=0)
    config['isotropy'] = np.max(median_spacing) / np.min(median_spacing)
    # We artificially inflate the third axes spacing so that the anisotropic defaults to the third axis if all other
    # axes are equal
    median_spacing[2] *= 1.01
    config['anisotropic_axis'] = int(np.argmax(median_spacing))
    median_spacing[2] /= 1.01

    config['target_spacing'] = median_spacing
    if config['isotropy'] >= 3:
        config['target_spacing'][config['anisotropic_axis']] = \
            np.percentile(train_spacing[config['anisotropic_axis'], :], 10)

    shapes = np.round(shapes * train_spacing / config['target_spacing'])
    config['depths'] = shapes[:, config['anisotropic_axis']]

    # This contains the shape after all preprocessing
    config['shape'] = np.max(shapes, axis=0)
    # move anisotropic axis to end
    config['shape'][2], config['shape'][config['anisotropic_axis']] \
        = config['shape'][config['anisotropic_axis']], config['shape'][2]

    compute_patch_size(config, memory_constraint)
    compute_batch_size(config, memory_constraint)

    # Convert numpy types to python types for JSON serialization
    for key, value in config.items():
        if type(value) == np.ndarray:
            config[key] = value.tolist()
        if type(value) == np.int64:
            config[key] = int(value)
        if type(value) == np.float64:
            config[key] = float(value)

    with open(join(config['path'], 'data.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    return config


def compute_patch_size(config, memory_constraint):
    """Computes the patch size given memory constraints"""
    # Limits on the batch size
    min_batch_size = 2
    channels = config['channels']
    patch_size = config['shape'].copy()
    while min_batch_size * 4 * channels * volume(patch_size) >= memory_constraint:
        largest_dim = np.argmax(patch_size)
        patch_size[largest_dim] = np.ceil(patch_size[largest_dim] / 2)

    config['patch_size'] = patch_size


def compute_batch_size(config, memory_constraint):
    """Computes the batch size given the patch size and memory constraints"""
    max_dataset_coverage_of_batch = 0.05
    patch_size = config['patch_size']
    channels = config['channels']
    shape = config['shape']
    max_batch_size = max_dataset_coverage_of_batch * config['num_train']

    # determine the batch size for both 2D and 3D networks
    batch_size_from_memory = np.floor(memory_constraint / (4 * channels * volume(patch_size)))
    config['batch_size3D'] = int(min(batch_size_from_memory, max_batch_size))

    slice_size = config['shape'][0] * config['shape'][1]
    batch_size_from_memory = np.floor(memory_constraint / (4 * channels * slice_size))
    max_batch_size *= config['shape'][2]
    config['batch_size2D'] = int(min(batch_size_from_memory, max_batch_size))

    minimum_patch_overlap = 20
    config['patches_along_each_axis'] = \
        np.ceil((shape - minimum_patch_overlap) / (patch_size - minimum_patch_overlap))

    # No overlap results in a 0 / 0 division, so suppress the warnings as this is intentional
    with np.errstate(divide='ignore'):
        config['patch_overlap'] = \
            np.floor((config['patches_along_each_axis'] * patch_size - shape) / (
                    config['patches_along_each_axis'] - 1))
        config['patch_overlap'] = np.nan_to_num(config['patch_overlap'])


def preprocess_dataset(path, name, memory_constraint):
    import nibabel as nib
    config = obtain_dataset_fingerprint(path, name, memory_constraint)
    for i in range(config['num_train']):
        label = nib.load(join(config['raw_path'], f'trLbl_{i}.nii.gz'))
        image = nib.load(join(config['raw_path'], f'trImg_{i}.nii.gz'))
        image, label = preprocess_img(config, image.get_fdata(), image.header.get_zooms(), label=label.get_fdata())
        np.savez_compressed(join(config['path'], f'train_{i}.npz'), image=image, label=label)

    for i in range(config['num_test']):
        image = nib.load(join(config['raw_path'], f'tsImg_{i}.nii.gz'))
        # No preprocessing here, but we do save the spacings so we can do preprocessing later
        np.savez_compressed(join(config['path'], f'test_{i}.npz'), image=image.get_fdata(),
                            spacing=image.header.get_zooms())


def preprocess_img(config, image, spacing, label=None):
    """
    Fully preprocess an entire image and label given the numpy arrays and the pixel spacings
    Takes in raw numpy images with no changes after loading the nifti file
    """
    image = np.expand_dims(image, 0) if image.ndim == 3 else np.moveaxis(image, -1, -0)
    image = np.swapaxes(image, 3, config['anisotropic_axis'] + 1)
    if label is not None:
        label = np.swapaxes(label, 2, config['anisotropic_axis'])

    image, label = crop(image, label)
    image = normalise(image, config)
    size = np.round(np.array(image.shape[1:]) * np.array(spacing[0:3]) / config['target_spacing'])
    image = resize_image(image, size, config)
    label = resize_label(label, size, config)
    return image, label


def resize(image, spacing, config, label=None):
    """Resizes both the image and label to match the target spacing"""
    new_size = np.round(image.shape[1:] * spacing / config['target_spacing'])
    return resize_image(image, new_size, config), resize_label(label, new_size, config)


def resize_label(label, size, config):
    """
    Resizes the label to the given size according to the isotropic rules

    :param label: input label
    :param size: tuple or list containing the size of the output
    :param config: data config containing at least the isotropy
    :return: a non one hot encoded label
    """
    if label is None:
        return None
    size = tuple(size)
    if np.all(size == label.shape):
        return label

    classes = config['n_classes']
    util.one_hot(label, classes, batch=False)
    if config['isotropy'] >= 3:
        first_size = (classes, size[0], size[1], label.shape[2])
        label = skimage.transform.resize(np.float(label), first_size, order=3)
        # Resize with nearest neighbor along 3rd dimension
        label = skimage.transform.resize(label, (classes,) + size, order=0)
        return np.argmax(label, dim=0)

    label = skimage.transform.resize(label.astype(float), (classes,) + size, order=3)
    return np.argmax(label, axis=0)


def resize_one_hot_label(label, size, config):
    """
    Resizes the label to the given size according to the isotropic rules

    Note: the returned label is not one hot encoded
    :param label: input label (as a one hot encoded label)
    :param size: tuple or list containing the size of the output
    :param config: data config containing at least the isotropy
    :return: a non one hot encoded label
    """
    if label is None:
        return None
    size = tuple(size)
    if np.all(size == label.shape[1:]):
        return np.argmax(label, axis=0)

    classes = label.shape[0]
    if config['isotropy'] >= 3:
        first_size = (classes, size[0], size[1], label.shape[2])
        label = skimage.transform.resize(np.float(label), first_size, order=3)
        # Resize with nearest neighbor along 3rd dimension
        label = skimage.transform.resize(label, (classes,) + size, order=0)
        return np.argmax(label, dim=0)

    label = skimage.transform.resize(label.astype(float), (classes,) + size, order=3)
    return np.argmax(label, axis=0)


def resize_image(image, size, config):
    """
    Resizes the image to the given size based on the isotropic rules

    :param image: The 4 dimensional image to resize
    :param size: The 3 dimensional size to resize it to as a tuple or list
    :param config: Data config containing at least the isotropy
    :return:
    """
    size = tuple(size)
    if np.all(size == image.shape[1:]):
        return image

    channels = image.shape[0]
    if config['isotropy'] >= 3:
        first_size = (channels, size[0], size[1], image.shape[2])
        image = skimage.transform.resize(image, first_size, order=3)
        # Resize with nearest neighbor along 3rd dimension
        return skimage.transform.resize(image, (channels,) + size, order=0)

    return skimage.transform.resize(image, (channels,) + size, order=3)


def normalise(data, config):
    """
    Channels that are of the CT modality type are clipped within the 0.5 and 99.5 percentiles and all channels are
    normalised to unit stdev and zero mean
    :param data: numpy array with channels as the first dimension
    :param config: config dictionary, needed for the modalities of each channel
    :return: normalised data
    """
    for i in range(data.shape[0]):
        if config['ct'][i]:
            percentile_99_5 = np.percentile(data[i], 99.5)
            percentile_0_5 = np.percentile(data[i], 0.5)
            data[i] = np.clip(data[i], percentile_0_5, percentile_99_5)
        mean = np.mean(data[i])
        std = np.std(data[i])
        data[i] = (data[i] - mean) / std

    return data
