import torch
from matplotlib import colors
import numpy as np
import imageio


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
        classes = np.unique(label)
    dims = [len(classes)] + list(label.shape)
    one_hot_encoding = np.uint8(np.zeros(dims))
    for i, c in enumerate(classes):
        one_hot_encoding[i, label == c] = True
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
        img[label[0] == 0] = label_colours[label[0] == 0]

    img = np.moveaxis(img, dim, 0)
    images = [img[i, :, :] for i in range(img.shape[0])]
    imageio.mimsave(file, images)
