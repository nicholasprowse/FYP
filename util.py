import json

import torch
from matplotlib import colors
import numpy as np
import imageio
import torch.nn as nn
from torch.nn.functional import pad
import os
from os.path import join
import matplotlib.pyplot as plt


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
        if type(label) == np.ndarray:
            n_classes = int(np.max(label)) + 1
        else:
            n_classes = int(torch.max(label)) + 1
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
    img -= np.min(img)
    maximum = np.max(img)
    if maximum > 0:
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
    score = [0] * (ground_truth.shape[1] + 1)
    for i in range(ground_truth.shape[1]):
        volume_sum = float(ground_truth[:, i].sum() + prediction[:, i].sum())
        if volume_sum == 0:
            score[i] = 1
        else:
            volume_intersect = float((ground_truth[:, i] * prediction[:, i]).sum())
            score[i] = 2 * volume_intersect / volume_sum

    score[-1] = sum(score[1:-1]) / (ground_truth.shape[1] - 1)
    return np.array(score)


def load_into_dict(training_dict, device, model='latest'):
    """
    Loads the given saved model into the dictionary, and returns the next training epoch that needs to be completed
    """
    if os.path.isfile(training_dict[f'{model}_model_path']):
        check_point = torch.load(training_dict[f'{model}_model_path'], map_location=device)
        if 'optimiser' in training_dict:
            training_dict['optimiser'].load_state_dict(check_point['optimiser_state_dict'])
        if 'model' in training_dict:
            training_dict['model'].load_state_dict(check_point['model_state_dict'])
        if 'lr_scheduler' in training_dict:
            training_dict['lr_scheduler'].load_state_dict(check_point['scheduler_state_dict'])
        training_dict['train_logger'] = check_point['train_loss']
        training_dict['validation_logger'] = check_point['valid_loss']
        training_dict['dice_logger'] = check_point['dice_score']
        training_dict['do_component_suppression'] = check_point['do_component_suppression']
        training_dict['completed_epochs'] = check_point['completed_epochs']
        training_dict['completed_mini_batches'] = check_point['completed_mini_batches']
        training_dict['best_performance'] = check_point['best_performance']
        print(f"{training_dict['dims']}D checkpoint loaded, starting from epoch:", check_point['completed_epochs']+1)
    else:
        # Raise Error if it does not exist
        print(f"{training_dict['dims']}D checkpoint does not exist, starting from scratch")


class TverskyLoss(nn.Module):
    def __init__(self, n_classes, alpha=0.7, gamma=0.75, weight=None):
        super(TverskyLoss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, prediction, ground_truth, softmax=False):
        if softmax:
            prediction = torch.softmax(prediction, dim=1)
        ground_truth = one_hot(ground_truth, n_classes=self.n_classes)
        weight = [1] * self.n_classes if self.weight is None else self.weight
        assert prediction.size() == ground_truth.size(), \
            'predict {} & target {} shape do not match'.format(prediction.size(), ground_truth.size())
        loss = 0.0
        for i in range(self.n_classes):
            tv = focal_tversky_loss(prediction[:, i], ground_truth[:, i], alpha=self.alpha, gamma=self.gamma)
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


def generate_plot(exp_numbers, param_of_interest):
    path = '/Users/nicholasprowse/Documents/Engineering/FYP/Results'
    val_loss = [[]] * len(exp_numbers)
    dice_score = [[]] * len(exp_numbers)
    param_vals = [0] * len(exp_numbers)
    for i, experiment in enumerate(exp_numbers):
        info = json.load(open(join(path, f'experiment_{experiment}/experiment.json')))
        param_vals[i] = info[param_of_interest]
        data = json.load(open(join(path, f'experiment_{experiment}/loss2D.json')))
        val_loss[i] = data['validation']
        dice_score[i] = [x[-1] for x in data['dice']]
        print(f"{param_of_interest}={param_vals[i]} - Loss: {min(val_loss[i]):.4f}", end='')
        min_dice_score = [max([x[j] for x in data['dice']]) for j in range(len(data['dice'][0]))]
        for j, score in enumerate(min_dice_score):
            print(f", {j}: {score:.3f}", end='')
        print()

    plt.figure()
    for loss in val_loss:
        plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(f"Validation loss for varying {param_of_interest}")
    plt.ylim([0, 0.5])
    plt.legend(param_vals)
    plt.savefig(join(path, f"{param_of_interest}_loss.pdf"))

    plt.figure()
    argmax = 0
    for i, score in enumerate(dice_score):
        if score[-1] > dice_score[argmax][-1]:
            argmax = i
        plt.plot(score)
    print(f'Best: {param_vals[argmax]}')
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title(f"Dice score for varying {param_of_interest}")
    plt.legend(param_vals)
    plt.savefig(join(path, f"{param_of_interest}_dice.pdf"))