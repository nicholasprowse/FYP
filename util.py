import torch
from matplotlib import colors
import numpy as np
import imageio
import torch.nn as nn
from torch.nn.functional import pad
import os


def one_hot(label, n_classes=None):
    """
    Takes a label formatted where each class is a different consecutive integer. Converts this into a
    one hot encoded label. If input has dimensions of (x1, x2, ..., xn) then output will have dimension
    (num_classes, x1, x2, ..., xn)
    :param label: label to be converted to one hot encoding
    :param n_classes: number of classes in the label
    :return:
    """
    if n_classes is None:
        n_classes = torch.max(label)
    dims = list(label.shape)
    dims.insert(1, n_classes)
    one_hot_encoding = torch.zeros(dims).int().to(label.device)
    for i in range(n_classes):
        one_hot_encoding[:, i, :, :][label == i] = 1
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
        # Uncomment this line to get just the label, without original image
        # img = np.zeros_like(img)
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
    def __init__(self, n_class):
        super(DiceLoss, self).__init__()
        self.n_class = n_class

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = one_hot(target, n_classes=self.n_class)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = _dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes


def generate_example_output(model, dataset, device, epoch, n_class, dims=2):
    """Saves a visualisation of the model output and the ground truth"""
    if dims == 2:
        model.eval()
        dataset.eval()
        img_slices = [0] * dataset.depths[0]
        lbl_slices = [0] * dataset.depths[0]
        for i in range(dataset.depths[0]):
            img_slices[i], lbl_slices[i] = dataset[i]
        image = torch.stack(img_slices)
        label = torch.stack(lbl_slices)
        label = one_hot(label, n_class).numpy()
        prediction = model(image.to(device)).detach()
        prediction = torch.argmax(prediction, dim=1)
        prediction = one_hot(prediction, n_class).cpu().numpy()
        image = image.numpy()
        if not os.path.exists('out'):
            os.mkdir('out')
        empty_label = np.zeros_like(label)
        empty_label[:, 0, :, :] = 1
        label = np.swapaxes(np.concatenate([empty_label, label, prediction], axis=2), 0, 1)
        image = np.tile(image[:, 0], (1, 3, 1))
        img2gif(image, 2, f"out/2D_epoch{epoch}.gif", label=label)
    else:
        model.eval()
        dataset.eval()
        image, label = dataset[0]
        image = image.unsqueeze(0)
        label = one_hot(label.unsqueeze(0), n_class).numpy().squeeze()
        prediction = model(image.to(device)).detach()
        prediction = torch.argmax(prediction, dim=1)
        prediction = one_hot(prediction, n_class).cpu().numpy().squeeze()
        image = image.numpy()
        if not os.path.exists('out'):
            os.mkdir('out')
        empty_label = np.zeros_like(label)
        empty_label[0, :, :, :] = 1
        label = np.concatenate([empty_label, label, prediction], axis=2)
        image = np.tile(image[0, 0], (1, 3, 1))
        img2gif(image, 2, f"out/3D_epoch{epoch}.gif", label=label)


def load_into_dict(training_dict):
    """
    Loads the given saved model into the dictionary, and returns the next training epoch that needs to be completed
    """
    if os.path.isfile(training_dict['save_path']):
        check_point = torch.load(training_dict['save_path'])
        training_dict['model'].load_state_dict(check_point['model_state_dict'])
        training_dict['optimiser'].load_state_dict(check_point['optimiser_state_dict'])
        training_dict['lr_scheduler'].load_state_dict(check_point['scheduler_state_dict'])
        training_dict['train_logger'] = check_point['train_loss']
        training_dict['validation_logger'] = check_point['valid_loss']
        print(f"{training_dict['dims']}D checkpoint loaded, starting from epoch:", check_point['epoch'])
        return check_point['epoch'] + 1
    else:
        # Raise Error if it does not exist
        print(f"{training_dict['dims']}D checkpoint does not exist, starting from scratch")
        return 0


class TverskyLoss(nn.Module):
    def __init__(self, n_classes):
        super(TverskyLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, prediction, ground_truth, weight=None, softmax=False):
        if softmax:
            prediction = torch.softmax(prediction, dim=1)
        ground_truth = one_hot(ground_truth, n_classes=self.n_classes)
        if weight is None:
            weight = [1] * self.n_classes
        assert prediction.size() == ground_truth.size(), \
            'predict {} & target {} shape do not match'.format(prediction.size(), ground_truth.size())
        loss = 0.0
        for i in range(1, self.n_classes):
            tv = focal_tversky_loss(prediction[:, i], ground_truth[:, i])
            loss += tv * weight[i]
        return loss / self.n_classes


def tversky(prediction, ground, smooth=1e-5, alpha=0.7):
    true_pos = torch.sum(ground * prediction)
    false_neg = torch.sum(ground * (1 - prediction))
    false_pos = torch.sum((1 - ground) * prediction)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(prediction, ground):
    return 1 - tversky(prediction, ground)


def focal_tversky_loss(prediction, ground, gamma=0.75):
    tv = tversky(prediction, ground)
    return torch.pow((1 - tv), gamma)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
