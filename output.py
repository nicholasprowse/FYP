import torch
import numpy as np
from os.path import join, exists
import os
import json
import preprocessing
from preprocessing import volume
from util import center_crop, img2gif, one_hot
from torch.nn.functional import pad
from dataset import index_to_patch_location
from postprocessing import remove_all_but_the_largest_connected_component


def generate_test_predictions(training_dict, data_config, device):
    with torch.no_grad():
        for i in range(data_config['num_test']):
            with np.load(join(data_config['raw_path'], f'test_{i}.npz')) as data:
                image = data['image']
                spacing = data['spacing']
            if training_dict['dims'] == 2:
                prediction = get_prediction_2d(image, spacing, training_dict, data_config, device)
            else:
                prediction = get_prediction_3d(image, spacing, training_dict, data_config, device)
            np.savez_compressed(join(training_dict['out_path'], f'pred_{i}.npz'), prediction=prediction)
            # prediction = one_hot(prediction, n_classes=data_config['n_classes'], batch=False)
            #
            # empty_label = np.zeros_like(prediction)
            # empty_label[0, :, :, :] = 1
            # label = np.concatenate([empty_label, prediction], axis=2)
            # for channel in range(data_config['channels']):
            #     img_channel = image if image.ndim == 3 else image[:, :, :, channel]
            #     img_channel = np.tile(img_channel, (1, 2, 1))
            #     img2gif(img_channel, 2, join(training_dict['out_path'], f'pred_{i}({channel}).gif'), label=label)


def get_prediction_3d(image, spacing, training_dict, data_config, device):
    img_shape = np.array(data_config['shape3d']).astype(int)
    patches_along_each_axis = np.array(data_config['patches_along_each_axis3d']).astype(int)
    patch_size = np.array(data_config['patch_size3d']).astype(int)
    channels = data_config['channels']
    model = training_dict['model']
    model.eval()
    batch_size = data_config['batch_size3d']
    patch_overlap = np.array(data_config['patch_overlap3d']).astype(int)

    cropped_size = preprocessing.get_nonzero_size(np.expand_dims(image, 0))
    crops = preprocessing.get_crops(np.expand_dims(image, 0))
    processed_image, _ = preprocessing.preprocess_img(data_config, image.copy(), spacing)
    processed_size = processed_image.shape
    processed_image = center_crop(processed_image, img_shape)
    # split the image into patches
    num_patches = int(volume(patches_along_each_axis))
    input_patches = np.zeros([num_patches, channels] + list(patch_size))
    for j in range(input_patches.shape[0]):
        patch_location = index_to_patch_location(j, patches_along_each_axis)
        lb = (np.minimum(patch_location * patch_size - patch_location * patches_along_each_axis,
                         img_shape - patch_size)).astype(int)
        ub = (lb + patch_size).astype(int)
        input_patches[j] = processed_image[:, lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]]

    input_patches = torch.from_numpy(input_patches).float().to(device)
    num_batches = int(np.ceil(num_patches / batch_size))
    output_patches = [torch.Tensor(0)] * num_batches
    for batch in range(num_batches):
        lower = batch * batch_size
        upper = min((batch + 1) * batch_size, num_patches)
        output_patches[batch] = model(input_patches[lower:upper])

    output_patches = torch.cat(output_patches, 0).detach().cpu().numpy()

    prediction = stitch_patches_together(output_patches, patches_along_each_axis, patch_overlap, img_shape)

    prediction = center_crop(prediction, processed_size[1:])
    prediction = preprocessing.resize_one_hot_label(prediction, cropped_size, data_config)
    prediction = pad(torch.from_numpy(prediction), crops).numpy()
    if training_dict['do_component_suppression']:
        prediction = remove_all_but_the_largest_connected_component(prediction, data_config['n_classes'])

    return np.moveaxis(prediction, -1, data_config['anisotropic_axis'])


def get_prediction_2d(image, spacing, training_dict, data_config, device):
    """
    Takes in a raw image and finds the prediction of the image. Performs preprocessing, postprocessing,
    separating slices into patches and restitching the labels together.
    :param device:
    :param data_config:
    :param training_dict:
    :param image: Unprocessed 3 or 4 dimensional image
    :param spacing: Voxel spacings of the 2 in plane dimensions
    :return: Multi hot encoded prediction of the image
    """
    img_shape = np.array(data_config['shape2d']).astype(int)
    model = training_dict['model']
    model.eval()
    batch_size = data_config['batch_size2d']
    patch_size = np.array(data_config['patch_size2d']).astype(int)
    patches_along_each_axis = np.array(data_config['patches_along_each_axis2d']).astype(int)
    patch_overlap = np.array(data_config['patch_overlap2d']).astype(int)
    cropped_size = preprocessing.get_nonzero_size(np.expand_dims(image, 0))
    crops = preprocessing.get_crops(np.expand_dims(image, 0))
    processed_image, _ = preprocessing.preprocess_img(data_config, image.copy(), spacing, dims=2)
    processed_size = processed_image.shape
    slices = np.moveaxis(processed_image, processed_image.ndim - 1, 0)
    slices = center_crop(slices, img_shape[:2])
    slices = torch.from_numpy(slices).float().to(device)

    # split the image into patches
    slice_batches = int(np.ceil(slices.shape[0] / batch_size))
    plane_batches = int(volume(patches_along_each_axis))
    patches = [np.array(0)] * plane_batches
    for plane_batch in range(plane_batches):
        patch_location = index_to_patch_location(plane_batch, patches_along_each_axis)
        lb = (np.minimum(patch_location * patch_size - patch_location * patches_along_each_axis,
                         img_shape - patch_size)).astype(int)
        ub = (lb + patch_size).astype(int)
        slice_outputs = [torch.Tensor(0)] * slice_batches
        for slice_batch in range(slice_batches):
            lower_slice = slice_batch * batch_size
            upper_slice = min((slice_batch + 1) * batch_size, slices.shape[0])

            batch = slices[lower_slice:upper_slice, :, lb[0]:ub[0], lb[1]:ub[1]]
            slice_outputs[slice_batch] = model(batch)

        patches[plane_batch] = torch.cat(slice_outputs, 0).detach().cpu().numpy()

    patches = np.stack(patches)
    patches = np.moveaxis(patches, 1, -1)
    output = stitch_patches_together(patches, patches_along_each_axis, patch_overlap, img_shape)
    output = center_crop(output, processed_size[1:])
    output = preprocessing.resize_one_hot_label(output, cropped_size[:3], data_config)
    output = pad(torch.from_numpy(output), crops[-6:]).numpy()
    if training_dict['do_component_suppression']:
        output = remove_all_but_the_largest_connected_component(output, data_config['n_classes'])

    return np.moveaxis(output, -1, data_config['anisotropic_axis'])


def stitch_patches_together(patches, patches_along_each_axis, patch_overlap, img_shape):
    """
    Stitches together a list of patches, whose ordering is consistent with that generated by index_to_patch_location.
    Only stitches along the n dimensions after the classes dimension (dimension 1), where n is the number of elements
    in patches_along_each_axis
    :param img_shape:
    :param patch_overlap:
    :param patches: list of patches
    :param patches_along_each_axis: Number of patches in each dimension. np array of length 2 or 3
    :return:
    """
    # Scale the boundaries of each patch with gaussian interpolation, so they can simply be added together
    # afterwards
    n = len(patches_along_each_axis)
    num_patches = patches.shape[0]
    num_classes = patches.shape[1]
    patch_size = np.array(patches.shape[2:2+n]).astype(int)
    for dim in range(n):
        # Create a gaussian kernel with stddev half of the overlap
        size = patch_overlap[dim] / 2
        base_kernel = np.arange(patch_overlap[dim])
        base_kernel = np.exp(-(base_kernel ** 2) / (2 * (size / 2) ** 2))
        # Normalise it, so when the other patch is added (with a kernel in the other direction),
        # they sum to 1
        base_kernel /= (base_kernel + base_kernel[::-1])
        # Do the same but for the final overlap which is different
        size = ((patches_along_each_axis * patch_size - img_shape -
                 (patches_along_each_axis - 2) * patch_overlap) / 2)[dim]
        final_kernel = np.arange(size)
        final_kernel = np.exp(-(final_kernel ** 2) / (2 * (size / 2) ** 2))
        final_kernel /= (final_kernel + final_kernel[::-1])
        # Shape to reshape kernel, so it runs along the dimension dim
        kernel_shape = [1] * 4
        kernel_shape[dim + 1] = patch_size[dim]

        for j in range(num_patches):
            patch_location = index_to_patch_location(j, patches_along_each_axis)
            # Every patch except the final 2 gets a base kernel on the right
            if patch_location[dim] < patches_along_each_axis[dim] - 2:
                # Add in ones to match the patch size
                kernel = np.concatenate([np.zeros(patch_size[dim] - len(base_kernel)), base_kernel])
                kernel.resize(kernel_shape)
                # Multiply it onto the patch
                patches[j] *= kernel
            # The second to last gets a final kernel on the right
            elif patch_location[dim] == patches_along_each_axis[dim] - 2:
                # Add in ones to match the patch size
                kernel = np.concatenate([np.zeros(patch_size[dim] - len(final_kernel)), final_kernel])
                kernel.resize(kernel_shape)
                # Multiply it onto the patch
                patches[j] *= kernel
            # Every patch except the first and last gets a base kernel on the left
            if patch_location[dim] < patches_along_each_axis[dim] - 1 or patch_location[dim] > 0:
                # Use final kernel on final patch, reverse it and pad with ones
                kernel = np.concatenate([base_kernel[::-1], np.zeros(patch_size[dim] - len(base_kernel))])
                kernel.resize(kernel_shape)
                # Multiply it onto the patch
                patches[j] *= kernel
            # The last gets a final kernel on the left
            elif patch_location[dim] == patches_along_each_axis[dim] - 2:
                # Use final kernel on final patch, reverse it and pad with ones
                kernel = np.concatenate([final_kernel[::-1], np.zeros(patch_size[dim] - len(final_kernel))])
                kernel.resize(kernel_shape)
                # Multiply it onto the patch
                patches[j] *= kernel

    # add all the patches together
    stitched = np.zeros([num_classes] + list(img_shape) + list(patches.shape[2+n:5]))
    for j in range(num_patches):
        patch_location = index_to_patch_location(j, patches_along_each_axis)
        lb = (np.minimum(patch_location * patch_size - patch_location * patches_along_each_axis,
                         img_shape - patch_size)).astype(int)
        ub = (lb + patch_size).astype(int)
        if n == 2:
            stitched[:, lb[0]:ub[0], lb[1]:ub[1], :] += patches[j]
        else:
            stitched[:, lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]] += patches[j]

    return stitched


def generate_training_output(training_dict, data_config, device, idx=0):
    """Saves a visualisation of the model output and the ground truth"""
    dims = training_dict['dims']

    data = np.load(join(data_config['raw_path'], f'train_{idx}.npz'))
    label = data['label']
    image = data['image']
    spacing = data['spacing']

    if training_dict['dims'] == 2:
        prediction = get_prediction_2d(image, spacing, training_dict, data_config, device)
    else:
        prediction = get_prediction_3d(image, spacing, training_dict, data_config, device)

    prediction = one_hot(prediction, data_config['n_classes'], batch=False)
    label = one_hot(label, data_config['n_classes'], batch=False)
    # Stitch the three images together and save a visualisation of it
    for channel in range(data_config['channels']):
        img_channel = image if image.ndim == 3 else image[:, :, :, channel]
        img2gif(img_channel, 2, join(training_dict['out_path'], f"{dims}D_train_gt_{idx}({channel}).gif"), label=label)
        img2gif(img_channel, 2, join(training_dict['out_path'],
                                     f"{dims}D_train_pred_{idx}({channel}).gif"), label=prediction)


def convert_output_to_nifti(task_name, task_id):
    """
    Converts the test outputs (npz files) generated by main.py into nifti images
    """
    import nibabel as nib
    liver_exclusions = [141, 156, 160, 161, 162, 164, 167, 182, 189, 190]
    out_path = 'out/predictions'
    if not exists(out_path):
        os.mkdir(out_path)
    if not exists(join(out_path, task_name)):
        os.mkdir(join(out_path, task_name))
    dataset_json = json.load(open(join('data/raw', task_name, 'dataset.json')))
    output_exists = exists(f'out/experiment_{task_id}')
    for i, path in enumerate(dataset_json['test']):
        name = path[path.rindex('/') + 1:]
        excluded = False
        for n in liver_exclusions:
            if 'Liver' in task_name and str(n) in name:
                excluded = True

        if excluded:
            continue

        nib_image = nib.load(join('data/raw', task_name, path))
        if not output_exists:
            prediction = np.zeros(nib_image.shape[:3]).astype(np.uint8)
            nifti = nib.Nifti1Image(prediction, nib_image.affine)
            nib.save(nifti, join(out_path, task_name, name))
        else:
            with np.load(join(f'out/experiment_{task_id}', f'pred_{i}.npz')) as data:
                prediction = data['prediction'].astype(np.uint8)
                nifti = nib.Nifti1Image(prediction, nib_image.affine)
                nib.save(nifti, join(out_path, task_name, name))
