import torch
from matplotlib import colors
import numpy as np
import imageio
import torch.nn as nn
from torch.nn.functional import pad


def one_hot(label, classes=None):
    """
    Takes a label formatted where each class is a different consecutive integer. Converts this into a
    one hot encoded label. If input has dimensions of (x1, x2, ..., xn) then output will have dimension
    (num_classes, x1, x2, ..., xn)
    :param label: label to be converted to one hot encoding
    :param classes: list of the values used to represent each class in the label
    :return:
    """
    if classes is None:
        classes = torch.unique(label)
    dims = list(label.shape)
    dims.insert(1, len(classes))
    one_hot_encoding = torch.zeros(dims).int()
    for i, c in enumerate(classes):
        one_hot_encoding[:, i, :, :][label == c] = 1
    return one_hot_encoding


def img2gif(img, dim, file, label=None):
    """
    Converts a 3D image into a gif file that traverses the 3D image along the given dimension
    If a label is supplied, the label will be overlaid on top of the image, where each class
    is a different color, and the background class is not displayed
    :param img: 3D pytorch or numpy array
    :param dim: The dimension that is traversed as time increases
    :param file: Save location for the file
    :param label: one-hot encoded label (height, width, depth, num_classes). This is optional
    """
    img = np.array(img)
    img = np.uint8(img * 255 / np.max(img))
    dim %= 3    # Force dim to be 0, 1 or 2
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
        # img[label[0] == 0] = label_colours[label[0] == 0]
        img = np.zeros_like(img)
        img[label[0] == 0] = label_colours[label[0] == 0]

    img = np.moveaxis(img, dim, 0)
    images = [img[i, :, :] for i in range(img.shape[0])]
    imageio.mimsave(file, images)


def _dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def _one_hot_encoder(self, input_tensor):
    tensor_list = []
    for i in range(self.n_classes):
        temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob.unsqueeze(1))
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


def center_crop(a, size):
    """Pads/crops and image appropriately, so the original image is centered, and has the given size"""
    padding_amount = tuple([int((size[i // 2] - a.shape[i // 2] + i % 2) // 2) for i in
                            range(2*len(size)-1, -1, -1)])
    return np.array(pad(torch.from_numpy(a), padding_amount))


class DiceLoss(nn.Module):
    def __init__(self, classes):
        super(DiceLoss, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = one_hot(target, classes=self.classes)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = _dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes
