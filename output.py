import torch
import numpy as np
from os.path import join
import preprocessing
from preprocessing import volume
from util import center_crop, img2gif, one_hot
from torch.nn.functional import pad
from dataset import index_to_patch_location
from postprocessing import remove_all_but_the_largest_connected_component


def generate_test_predictions(training_dict, data_config, device):
    if training_dict['dims'] == 2:
        generate_test_predictions2d(training_dict, data_config, device)
    else:
        generate_test_predictions3d(training_dict, data_config, device)


def generate_test_predictions2d(training_dict, data_config, device):
    img_shape = np.array(data_config['shape']).astype(int)
    model = training_dict['model']
    batch_size = data_config['batch_size3D']
    classes = data_config['n_classes']
    for i in range(data_config['num_test']):
        with np.load(join(data_config['path'], f'test_{i}.npz')) as data:
            image = data['image']
            spacing = data['spacing']
        cropped_size = preprocessing.get_nonzero_size(np.expand_dims(image, 0))
        crops = preprocessing.get_crops(np.expand_dims(image, 0))
        processed_image, _ = preprocessing.preprocess_img(data_config, image.copy(), spacing)
        slices = np.moveaxis(processed_image, processed_image.ndim-1, 0)
        processed_size = slices.shape
        slices = center_crop(slices, img_shape[1:])
        slices = torch.from_numpy(slices).float().to(device)
        num_batches = int(np.ceil(slices.shape[0] / batch_size))
        outputs = [torch.Tensor(0)] * num_batches
        for batch in range(num_batches):
            lower = batch * batch_size
            upper = min((batch + 1) * batch_size, slices.shape[0])
            outputs[batch] = model(slices[lower:upper])

        outputs = torch.cat(outputs, 0).detach().cpu().numpy()
        outputs = center_crop(outputs, processed_size[2:])
        outputs = np.moveaxis(outputs, 0, outputs.ndim-1)
        outputs = preprocessing.resize_one_hot_label(outputs, cropped_size, data_config)
        outputs = pad(torch.from_numpy(outputs), crops).numpy()
        if training_dict['do_component_suppression']:
            outputs = remove_all_but_the_largest_connected_component(outputs, data_config['n_classes'])
        np.savez_compressed(join(training_dict['out_path'], f'pred_{i}.npz'), prediction=outputs)
        outputs = one_hot(outputs, n_classes=classes, batch=False)
        img2gif(image, 2, join(training_dict['out_path'], f'pred_{i}.gif'), label=outputs)


def generate_test_predictions3d(training_dict, data_config, device):
    img_shape = np.array(data_config['shape']).astype(int)
    patches_along_each_axis = np.array(data_config['patches_along_each_axis']).astype(int)
    patch_size = np.array(data_config['patch_size']).astype(int)
    channels = data_config['channels']
    model = training_dict['model']
    batch_size = data_config['batch_size3D']
    classes = data_config['n_classes']
    patch_overlap = np.array(data_config['patch_overlap']).astype(int)
    for i in range(data_config['num_test']):
        with np.load(join(data_config['path'], f'test_{i}.npz')) as data:
            image = data['image']
            spacing = data['spacing']
        cropped_size = preprocessing.get_nonzero_size(np.expand_dims(image, 0))
        crops = preprocessing.get_crops(np.expand_dims(image, 0))
        processed_image, _ = preprocessing.preprocess_img(data_config, image.copy(), spacing)
        processed_size = processed_image.shape
        processed_image = center_crop(processed_image, img_shape)
        # split the image into patches
        num_patches = int(volume(patches_along_each_axis))
        patches = np.zeros([num_patches, channels] + list(patch_size))
        for j in range(patches.shape[0]):
            patch_location = index_to_patch_location(j, patches_along_each_axis)
            lb = (np.minimum(patch_location * patch_size - patch_location * patches_along_each_axis,
                             img_shape - patch_size)).astype(int)
            ub = (lb + patch_size).astype(int)
            patches[j] = processed_image[:, lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]]

        patches = torch.from_numpy(patches).float().to(device)
        num_batches = int(np.ceil(num_patches / batch_size))
        outputs = [torch.Tensor(0)] * num_batches
        for batch in range(num_batches):
            lower = batch * batch_size
            upper = min((batch + 1) * batch_size, num_patches)
            outputs[batch] = model(patches[lower:upper])

        outputs = torch.cat(outputs, 0).detach().cpu().numpy()

        # Scale the boundaries of each patch with gaussian interpolation, so they can simply be added together
        # afterwards
        for dim in range(3):
            for j in range(num_patches):
                patch_location = index_to_patch_location(j, patches_along_each_axis)
                # Create a gaussian kernel with stddev half of the overlap
                size = patch_overlap[dim] / 2
                base_kernel = np.arange(patch_overlap[dim])
                base_kernel = np.exp(-(base_kernel ** 2) / (2 * (size / 2) ** 2))
                # Normalise it, so when the other patch is added (with a kernel in the other direction),
                # they sum to 1
                base_kernel /= (base_kernel + base_kernel[::-1])
                # Do the same but for the final overlap which is different
                size = ((num_patches * patch_size - img_shape - (num_patches - 2) * patch_overlap) / 2)[dim]
                final_kernel = np.arange(size)
                final_kernel = np.exp(-(final_kernel ** 2) / (2 * (size / 2) ** 2))
                final_kernel /= (final_kernel + final_kernel[::-1])
                # Shape to reshape kernel, so it runs along the dimension dim
                kernel_shape = [1] * 3
                kernel_shape[dim] = patch_size[dim]
                # Every patch except the final 2 gets a base kernel on the right
                if patch_location[dim] < patches_along_each_axis[dim] - 2:
                    # Add in ones to match the patch size
                    kernel = np.concatenate([np.zeros(patch_size[dim] - len(base_kernel)), base_kernel])
                    kernel.resize(kernel_shape)
                    # Multiply it onto the patch
                    outputs[j] *= kernel
                # The second to last gets a final kernel on the right
                elif patch_location[dim] == patches_along_each_axis[dim] - 2:
                    # Add in ones to match the patch size
                    kernel = np.concatenate([np.zeros(patch_size[dim] - len(final_kernel)), final_kernel])
                    kernel.resize(kernel_shape)
                    # Multiply it onto the patch
                    outputs[j] *= kernel
                # Every patch except the first and last gets a base kernel on the left
                if patch_location[dim] < patches_along_each_axis[dim] - 1 or patch_location[dim] > 0:
                    # Use final kernel on final patch, reverse it and pad with ones
                    kernel = np.concatenate([base_kernel[::-1], np.zeros(patch_size[dim] - len(base_kernel))])
                    kernel.resize(kernel_shape)
                    # Multiply it onto the patch
                    outputs[j] *= kernel
                # The last gets a final kernel on the left
                elif patch_location[dim] == patches_along_each_axis[dim] - 2:
                    # Use final kernel on final patch, reverse it and pad with ones
                    kernel = np.concatenate([final_kernel[::-1], np.zeros(patch_size[dim] - len(final_kernel))])
                    kernel.resize(kernel_shape)
                    # Multiply it onto the patch
                    outputs[j] *= kernel

        # add all the patches together
        out_prediction = np.zeros([classes] + list(processed_image.shape[1:]))
        for j in range(num_patches):
            patch_location = index_to_patch_location(j, patches_along_each_axis)
            lb = (np.minimum(patch_location * patch_size - patch_location * patches_along_each_axis,
                             img_shape - patch_size)).astype(int)
            ub = (lb + patch_size).astype(int)
            out_prediction[:, lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]] += outputs[j]

        out_prediction = center_crop(out_prediction, processed_size[1:])
        out_prediction = preprocessing.resize_one_hot_label(out_prediction, cropped_size, data_config)
        out_prediction = pad(torch.from_numpy(out_prediction), crops).numpy()
        if training_dict['do_component_suppression']:
            out_prediction = remove_all_but_the_largest_connected_component(out_prediction, data_config['n_classes'])
        np.savez_compressed(join(training_dict['out_path'], f'pred_{i}.npz'), prediction=out_prediction)
        out_prediction = one_hot(out_prediction, n_classes=classes, batch=False)
        img2gif(image, 2, join(training_dict['out_path'], f'pred_{i}.gif'), label=out_prediction)


def convert_output_to_nifti(out_path, data_config, folder_name, file_name):
    import nibabel as nib
    for i in range(data_config['num_test']):
        with np.load(join(out_path, f'pred_{i}.npz')) as data:
            image = data['prediction']
            nifti = nib.Nifti1Image(image, np.eye(4))
            nib.save(nifti, join(out_path, f'pred_{i}.nii.gz'))
