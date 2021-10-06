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
        shape[dim - 1] = np.max(idx[dim]) - np.min(idx[dim]) + 1
    return tuple(shape)


def volume(shape):
    """returns the total number of voxels in an image"""
    return reduce(lambda a, b: a * b, shape)


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
        image = np.expand_dims(image, 0) if image.ndim == 3 else np.moveaxis(image, -1, 0)
        shapes[i, :] = get_nonzero_size(image)
        config['channels'] = image.shape[0]

        label = nib.load(join(dataset_path, f'trLbl_{i}.nii.gz')).get_fdata()
        unique, counts = np.unique(label, return_counts=True)
        for j, clazz in enumerate(unique):
            class_frequency[int(clazz)] += counts[j] / volume(label.shape)

    class_frequency /= config['num_train']
    config['class_frequency'] = class_frequency
    # Calculate various dataset fingerprints based on this information
    median_spacing = np.median(train_spacing, axis=0)
    config['isotropy'] = np.max(median_spacing) / np.min(median_spacing)
    # We artificially inflate the third axes spacing so that the anisotropic defaults to the third axis if all other
    # axes are equal
    median_spacing[2] *= 1.01
    aia = int(np.argmax(median_spacing))
    median_spacing[2] /= 1.01
    config['anisotropic_axis'] = aia

    config['num_slices'] = np.sum(shapes[:, aia])

    config['target_spacing3d'] = median_spacing.copy()
    if config['isotropy'] >= 3:
        config['target_spacing3d'][aia] = np.percentile(train_spacing[:, aia], 10)
        voxel_increase = volume(config['target_spacing3d'] / median_spacing)
        config['target_spacing3d'] /= voxel_increase ** (1/3)

    modified_shapes = np.round(shapes * train_spacing / config['target_spacing3d'])
    config['shape3d'] = np.max(modified_shapes, axis=0)

    # move anisotropic axis to end
    config['shape3d'][[2, aia]] = config['shape3d'][[aia, 2]]
    config['target_spacing3d'][[2, aia]] = config['target_spacing3d'][[aia, 2]]

    # Get 2d spacing and shapes
    median_spacing[[2, aia]] = median_spacing[[aia, 2]]
    config['target_spacing2d'] = median_spacing[:2]
    config['shape2d'] = np.max(shapes, axis=0)
    config['shape2d'][[2, aia]] = config['shape2d'][[aia, 2]]
    config['shape2d'] = config['shape2d'][:2]

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


min_batch_size = 2


def compute_patch_size(config, memory_constraint):
    """Computes the patch size given memory constraints"""
    # Limits on the batch size
    channels = config['channels']

    patch_size3d = config['shape3d'].copy()
    while min_batch_size * 4 * channels * volume(patch_size3d) >= memory_constraint:
        largest_dim = np.argmax(patch_size3d)
        patch_size3d[largest_dim] = np.ceil(patch_size3d[largest_dim] / 2)
    config['patch_size3d'] = patch_size3d

    patch_size2d = config['shape2d'].copy()
    while min_batch_size * 4 * channels * volume(patch_size2d) >= memory_constraint:
        largest_dim = np.argmax(patch_size2d)
        patch_size2d[largest_dim] = np.ceil(patch_size2d[largest_dim] / 2)
    config['patch_size2d'] = patch_size2d


# If there is no overlap this results in a 0 / 0 division,  which is intentional, so we suppress the warnings
@np.errstate(divide='ignore', invalid='ignore')
def compute_batch_size(config, memory_constraint):
    """Computes the batch size given the patch size and memory constraints"""
    max_dataset_coverage_of_batch = 0.05
    minimum_patch_overlap = 20
    channels = config['channels']

    # determine the batch size for 3D network
    patch_size3d = config['patch_size3d']
    shape3d = config['shape3d']
    max_batch_size = max(max_dataset_coverage_of_batch * config['num_train'], min_batch_size)
    batch_size_from_memory = np.floor(memory_constraint / (4 * channels * volume(patch_size3d)))
    config['batch_size3d'] = int(min(batch_size_from_memory, max_batch_size))

    # Determine the number of patches along each axis and how much they will overlap
    config['patches_along_each_axis3d'] = \
        np.ceil((shape3d - minimum_patch_overlap) / (patch_size3d - minimum_patch_overlap))
    config['patch_overlap3d'] = \
        np.floor((config['patches_along_each_axis3d'] * patch_size3d - shape3d) / (
                config['patches_along_each_axis3d'] - 1))
    config['patch_overlap3d'] = np.nan_to_num(config['patch_overlap3d'])

    # determine the batch size for 2D network
    patch_size2d = config['patch_size2d']
    shape2d = config['shape2d']
    max_batch_size = max(max_dataset_coverage_of_batch * config['num_slices'], min_batch_size)
    batch_size_from_memory = np.floor(memory_constraint / (4 * channels * volume(patch_size2d)))
    config['batch_size2d'] = int(min(batch_size_from_memory, max_batch_size))
    config['patches_along_each_axis2d'] = \
        np.ceil((shape2d - minimum_patch_overlap) / (patch_size2d - minimum_patch_overlap))
    config['patch_overlap2d'] = \
        np.floor((config['patches_along_each_axis2d'] * patch_size2d - shape2d) / (
                config['patches_along_each_axis2d'] - 1))
    config['patch_overlap2d'] = np.nan_to_num(config['patch_overlap2d'])


def preprocess_dataset(path, name, memory_constraint):
    import nibabel as nib
    config = obtain_dataset_fingerprint(path, name, memory_constraint)
    print('Dataset fingerprint obtained')
    slice_no = 0
    for i in range(config['num_train']):
        label = nib.load(join(config['raw_path'], f'trLbl_{i}.nii.gz'))
        image = nib.load(join(config['raw_path'], f'trImg_{i}.nii.gz'))
        image2d, label2d = \
            preprocess_img(config, image.get_fdata(), image.header.get_zooms(), dims=2, label=label.get_fdata())
        image3d, label3d = \
            preprocess_img(config, image.get_fdata(), image.header.get_zooms(), dims=3, label=label.get_fdata())
        np.savez_compressed(join(config['path'], f'train_{i}.npz'), image=image3d, label=label3d)
        for j in range(image2d.shape[-1]):
            np.savez_compressed(join(config['path'], f'slice_{slice_no}.npz'),
                                image=image2d[:, :, :, j], label=label2d[:, :, j])
            slice_no += 1
        print(f"Processed {i}/{config['num_train']} training images")

    for i in range(config['num_test']):
        image = nib.load(join(config['raw_path'], f'tsImg_{i}.nii.gz'))
        # No preprocessing here, but we do save the spacings so we can do preprocessing later
        np.savez_compressed(join(config['path'], f'test_{i}.npz'), image=image.get_fdata(),
                            spacing=image.header.get_zooms())
        print(f"Processed {i}/{config['num_test']} testing images")


def preprocess_img(config, image, spacing, dims=3, label=None):
    """
    Fully preprocess an entire image and label given the numpy arrays and the pixel spacings
    Takes in raw numpy images with no changes after loading the nifti file

    dims determines how the image is resized, NOT the number of dimensions of the image. The image should always be 3
    or four dimensions with a 3 dimensional label. If dims is 2, then the image is only resized along the in plane axis
    for use in the 2D model. If dims is 3, then the image is resized along all dimensions for use in the 3D model
    """
    image = np.expand_dims(image, 0) if image.ndim == 3 else np.moveaxis(image, -1, 0)
    image = np.swapaxes(image, 3, config['anisotropic_axis'] + 1)
    spacing = np.array(spacing[0:3])
    spacing[[config['anisotropic_axis'], 2]] = spacing[[2, config['anisotropic_axis']]]

    if label is not None:
        label = np.swapaxes(label, 2, config['anisotropic_axis'])

    image, label = crop(image, label)
    image = normalise(image, config)
    size = np.round(np.array(image.shape[1:dims+1]) * spacing[:dims] / config[f'target_spacing{dims}d'][:dims])
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
    Returns 2 labels, the first is one that has been resized for the 3D model and the second is one that has been
    resized for the 2D model
    :param label: input multi hot encoded label
    :param size: tuple or list containing the size of the output
    :param config: data config containing at least the isotropy
    :return: a multi hot encoded label
    """
    if label is None:
        return None
    size = tuple([int(i) for i in size])
    if size == label.shape[:len(size)]:
        return label.astype(np.uint8)

    one_hot = util.one_hot(label, config['n_classes'], batch=False)
    return resize_one_hot_label(one_hot, size, config)


def resize_one_hot_label(label, size, config):
    """
    Resizes the label to the given size according to the isotropic rules
    If the provided size is only 2 dimensional, then only the first 2 spatial dimensions are resized

    Note: the returned label is not one hot encoded
    :param label: input label (as a one hot encoded label)
    :param size: tuple or list containing the size of the output
    :param config: data config containing at least the isotropy
    :return: a multi hot encoded label
    """
    if label is None:
        return None
    size = tuple([int(i) for i in size])
    if size == label.shape[1:1+len(size)]:
        multi_hot = np.argmax(label, axis=0)
        return multi_hot.astype(np.uint8)

    classes = label.shape[0]
    label = label.astype(float)
    # 2D label is only resized in plane, while 3D label is resized along all dimensions
    label2d = np.zeros((classes, size[0], size[1], label.shape[3]))
    label3d = np.zeros((classes,) + size)

    # If we are only resizing in 2 dimensions, or the image is isotropic then resize in plane
    if len(size) == 2 or config['isotropy'] >= 3:
        for clazz in range(classes):
            # First order for in plane resizing
            for j in range(label.shape[-1]):
                label2d[clazz, :, :, j] = skimage.transform.resize(label[clazz, :, :, j], size[:2], order=1)

        # If we entered this if statement due to 2d resizing, then we are done
        if len(size) == 2:
            return np.argmax(label2d, axis=0).astype(np.uint8)

        # Otherwise, we entered due to anisotropy, so resize along 3rd dimension
        for clazz in range(classes):
            # Resize with nearest neighbor along 3rd dimension
            for j in range(size[0]):
                label3d[clazz, j, :, :] = \
                    skimage.transform.resize(label2d[clazz, j, :, :], size[1:], order=0)

        return np.argmax(label3d, axis=0).astype(np.uint8)

    # Otherwise, we are simply resizing all dimensions at once
    for clazz in range(classes):
        label3d[clazz] = skimage.transform.resize(label[clazz], size, order=1)
    return np.argmax(label3d, axis=0).astype(np.uint8)


def resize_image(image, size, config):
    """
    Resizes the image to the given size based on the isotropic rules
    If the provided size is only 2 dimensional, then only the first 2 spatial dimensions are resized

    :param image: The 4 dimensional image to resize
    :param size: The 2 or 3 dimensional size to resize it to as a tuple or list
    :param config: Data config containing at least the isotropy
    :return:
    """
    size = tuple([int(i) for i in size])
    if size == image.shape[1:1+len(size)]:
        return image.astype(np.float32)

    channels = image.shape[0]
    # 2D image is only resized in plane, while 3D image is resized along all dimensions
    image3d = np.zeros((channels,) + size)
    image2d = np.zeros((channels, size[0], size[1], image.shape[3]))

    # If we are only resizing in 2 dimensions, or the image is isotropic then resize in plane
    if len(size) == 2 or config['isotropy'] >= 3:
        for channel in range(channels):
            # 3rd order interpolation for in plane resizing
            for j in range(image.shape[-1]):
                image2d[channel, :, :, j] = skimage.transform.resize(image[channel, :, :, j], size[:2], order=3)

        # If we entered this if statement due to 2d resizing, then we are done
        if len(size) == 2:
            return image2d.astype(np.float32)

        # Otherwise, we entered due to anisotropy, so resize along 3rd dimension
        for channel in range(channels):
            # Resize with nearest neighbor along 3rd dimension
            for j in range(size[0]):
                image3d[channel, j, :, :] = \
                    skimage.transform.resize(image2d[channel, j, :, :], size[1:], order=0)

        return image3d.astype(np.float32)

    # Otherwise, we are simply resizing all dimensions at once
    for channel in range(channels):
        image3d[channel] = skimage.transform.resize(image[channel].astype(float), size, order=3)
    return image3d.astype(np.float32)


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
