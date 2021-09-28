import torch
from matplotlib import colors
import numpy as np
import imageio
import torch.nn as nn
from torch.nn.functional import pad
import os
from os.path import join
import preprocessing


def one_hot(label, n_classes=None, batch=True):
    """
    Takes a label formatted where each class is a different consecutive integer. Converts this into a
    one hot encoded label. If input has dimensions of (x1, x2, ..., xn) then output will have dimension
    (num_classes, x1, x2, ..., xn)
    :param label: label to be converted to one hot encoding
    :param n_classes: number of classes in the label
    :param batch: whether to treat this as a batch of labels or a single image. For a single image the class dimension
    is inserted at dimension 0, while for a batch it is inserted at dimension 1 so that the batches are still the first
    dimension
    :return:
    """
    if n_classes is None:
        n_classes = torch.max(label)
    dims = list(label.shape)
    dims.insert(1 if batch else 0, n_classes)
    if type(label) == np.ndarray:
        one_hot_encoding = np.zeros(dims)
    else:
        one_hot_encoding = torch.zeros(dims).int().to(label.device)
    for i in range(n_classes):
        if batch:
            one_hot_encoding[:, i][label == i] = 1
        else:
            one_hot_encoding[i][label == i] = 1
    return one_hot_encoding


def img2gif(img, dim, file, label=None):
    """
    Converts a 3D image into a gif file that traverses the 3D image along the given dimension
    If a label is supplied, the label will be overlaid on top of the image, where each class
    is a different color, and the background class is not displayed
    :param img: 3D pytorch or numpy array
    :param dim: The dimension that is traversed as time increases
    :param file: Save location for the file
    :param label: one-hot encoded label (num_classes, height, width, depth). This is optional
    """
    img = np.array(img)
    img = np.uint8(img * 255 / np.max(img))
    dim %= 3  # Force dim to be 0, 1 or 2
    img = np.expand_dims(img, 3)
    img = np.tile(img, [1, 1, 1, 3])

    if label is not None:
        label = np.array(label)
        num_labels = label.shape[0] - 1
        label_colours = np.ones(img.shape)
        label_colours[:, :, :, 0] = 0
        for i in range(num_labels):
            label_colours[:, :, :, 0] += i * label[i + 1] / num_labels

        label_colours = 255 * colors.hsv_to_rgb(label_colours)
        # Uncomment this line to get just the label, without original image
        # img = np.zeros_like(img)
        img[label[0] == 0] = label_colours[label[0] == 0]

    img = np.moveaxis(img, dim, 0)
    images = [img[i, :, :] for i in range(img.shape[0])]
    imageio.mimsave(file, images)


def center_crop(a, size):
    """
    Pads/crops and image appropriately, so the original image is centered, and has the given size. If size has length
    N, then only the last N dimensions are cropped/padded.
    This operation is reversible. i.e. If a has size X and is padded to a size of Y, then cropped back to its original
    size X, then a will be unchanged.
    """
    total_padding = []
    for dim in range(-1, -len(size)-1, -1):
        padding = [abs(size[dim] - a.shape[dim]) // 2] * 2
        if (size[dim] - a.shape[dim]) % 2 == 1:
            padding[1] += 1
        if size[dim] - a.shape[dim] < 0:
            padding = [-padding[0], -padding[1]]
        total_padding += padding

    total_padding = [int(i) for i in total_padding]
    return np.array(pad(torch.from_numpy(a), total_padding))


def compute_dice_score(ground_truth, prediction):
    """
    Compute soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.

    Args:
      ground_truth: 4-dim tensor. The ground truth mask.
      prediction: 4-dim tensor. The predicted segmentation.

    Returns:
      the dice coefficient as float. If both masks are empty, the result is NaN

    Adapted from Medical Segmentation Decathlon: http://medicaldecathlon.com/
    """
    n_classes = prediction.shape[1]
    prediction = torch.argmax(prediction, dim=1)
    prediction = one_hot(prediction, n_classes=n_classes, batch=True)
    ground_truth = one_hot(ground_truth, n_classes=n_classes, batch=True)
    score = 0
    for i in range(1, ground_truth.shape[1]):
        volume_sum = float(ground_truth[:, i].sum() + prediction[:, i].sum())
        if volume_sum == 0:
            score += 1
        else:
            volume_intersect = float((ground_truth[:, i] * prediction[:, i]).sum())
            score += 2 * volume_intersect / volume_sum

    return score / ground_truth.shape[1]


def generate_example_output(training_dict, data_config, device, epoch):
    """Saves a visualisation of the model output and the ground truth"""
    import nibabel as nib
    dataset = training_dict['dataset']
    dims = training_dict['dims']
    n_classes = data_config['n_classes']
    training_dict['model'].eval()
    dataset.eval()
    # Load image and metadata
    with np.load(join(data_config['path'], f'train_0.npz')) as data:
        image = center_crop(data['image'], [data_config['channels']] + data_config['shape'])

    raw_image = nib.load(join(data_config['raw_path'], f'trImg_0.nii.gz')).get_fdata()
    raw_label = nib.load(join(data_config['raw_path'], f'trLbl_0.nii.gz')).get_fdata()
    raw_image = np.expand_dims(raw_image, 0) if raw_image.ndim == 3 else np.moveaxis(raw_image, -1, -0)
    raw_label = one_hot(raw_label, n_classes, batch=False)
    # Prepare image for the model
    image = np.moveaxis(image, 3, 0) if dims == 2 else np.expand_dims(image, 0)
    image = torch.from_numpy(image).float()
    # Make sure anisotropic axis is correct
    raw_image = np.swapaxes(raw_image, 3, data_config['anisotropic_axis'] + 1)
    raw_label = np.swapaxes(raw_label, 3, data_config['anisotropic_axis'] + 1)
    # Generate the prediction with the model
    prediction = training_dict['model'](image.to(device)).detach().cpu().numpy()
    prediction = np.moveaxis(prediction, 0, 3) if dims == 2 else np.squeeze(prediction, 0)
    image = np.moveaxis(image.numpy(), 0, 3) if dims == 2 else np.squeeze(image.numpy(), 0)
    # Remove padding added in the dataloader
    image, prediction = preprocessing.crop(image, label=prediction)
    # Resize to original spacing
    cropped_size = preprocessing.get_nonzero_size(raw_image)
    prediction = preprocessing.resize_label(prediction, cropped_size, data_config)
    # Add back in padding that was cropped out of original image
    prediction = center_crop(prediction, raw_label.shape[1:])
    prediction = one_hot(prediction, n_classes, batch=False)
    # Stitch the three images together and save a visualisation of it
    empty_label = np.zeros_like(raw_label)
    empty_label[0, :, :, :] = 1
    label = np.concatenate([empty_label, raw_label, prediction], axis=2)
    image = np.tile(raw_image[0], (1, 3, 1))

    img2gif(image, 2, join(training_dict['out_path'], f"{dims}D_epoch{epoch}.gif"), label=label)


def load_into_dict(training_dict, device):
    """
    Loads the given saved model into the dictionary, and returns the next training epoch that needs to be completed
    """
    if os.path.isfile(training_dict['model_path']):
        check_point = torch.load(training_dict['model_path'], map_location=device)
        training_dict['optimiser'].load_state_dict(check_point['optimiser_state_dict'])
        training_dict['model'].load_state_dict(check_point['model_state_dict'])
        training_dict['lr_scheduler'].load_state_dict(check_point['scheduler_state_dict'])
        training_dict['train_logger'] = check_point['train_loss']
        training_dict['validation_logger'] = check_point['valid_loss']
        training_dict['dice_logger'] = check_point['dice_score']
        training_dict['do_component_suppression'] = check_point['do_component_suppression']
        print(f"{training_dict['dims']}D checkpoint loaded, starting from epoch:", check_point['epoch'])
        return check_point['epoch'] + 1
    else:
        # Raise Error if it does not exist
        print(f"{training_dict['dims']}D checkpoint does not exist, starting from scratch")
        return 0


class TverskyLoss(nn.Module):
    def __init__(self, n_classes, alpha=None, eight=None):
        super(TverskyLoss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight
        self.alpha = alpha

    def forward(self, prediction, ground_truth, softmax=False):
        if softmax:
            prediction = torch.softmax(prediction, dim=1)
        ground_truth = one_hot(ground_truth, n_classes=self.n_classes)
        weight = [1] * self.n_classes if self.weight is None else self.weight
        assert prediction.size() == ground_truth.size(), \
            'predict {} & target {} shape do not match'.format(prediction.size(), ground_truth.size())
        loss = 0.0
        for i in range(self.n_classes):
            tv = focal_tversky_loss(prediction[:, i], ground_truth[:, i], alpha=self.alpha)
            loss += tv * weight[i]
        return loss / self.n_classes


def tversky(prediction, ground, smooth=1e-5, alpha=0.7):
    true_pos = torch.sum(ground * prediction)
    false_neg = torch.sum(ground * (1 - prediction))
    false_pos = torch.sum((1 - ground) * prediction)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(prediction, ground):
    return 1 - tversky(prediction, ground)


def focal_tversky_loss(prediction, ground, alpha=0.7, gamma=0.75):
    tv = tversky(prediction, ground, alpha=alpha)
    return torch.pow((1 - tv), gamma)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
