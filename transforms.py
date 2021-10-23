import torch
import numpy as np
from scipy import ndimage
from util import center_crop
# ALL TRANSFORMS TAKE NUMPY ARRAYS - NOT TORCH TENSORS
# HOWEVER, RANDOM NUMBERS NEED TO BE FROM TORCH
# (see https://pytorch.org/docs/stable/notes/faq.html#my-data-loader-workers-return-identical-random-numbers)


class RandomFlip:
    def __init__(self):
        pass

    def __call__(self, sample):
        img, lbl = sample
        axes = np.squeeze(np.array(torch.nonzero(torch.randint(2, (img.ndim-1,)), as_tuple=False)))
        img = np.flip(img, axes+1)  # Add one due to channels axis
        lbl = np.flip(lbl, axes)
        return img, lbl


class RandomRotateAndScale:
    def __init__(self, config, prob_rotate=0.2, prob_scale=0.2):
        patch_size = np.array(config['patch_size3d'])
        self.anisotropic = np.max(patch_size) >= 3 * np.min(patch_size)
        self.anisotropic_axis = np.argmax(patch_size)
        self.prob_scale = prob_scale
        self.prob_rotate = prob_rotate

    def __call__(self, sample):
        img, lbl = sample
        random = torch.rand(2)
        size = img.shape
        if random[1] <= self.prob_scale:
            zoom_amount = float(torch.rand(1) * 0.7 + 0.7)
            img = ndimage.zoom(img, zoom_amount)
            img = center_crop(img, size)
            lbl = ndimage.zoom(lbl, zoom_amount, order=0)
            lbl = center_crop(lbl, size[1:])

        if random[0] <= self.prob_rotate:
            if img.ndim == 4:
                angle = (torch.rand(3)-0.5)*60
                if self.anisotropic:
                    angle /= 2
                    angle[self.anisotropic_axis] *= 2

                img = ndimage.rotate(img, angle[0], (2, 3), reshape=False)
                img = ndimage.rotate(img, angle[1], (1, 3), reshape=False)
                img = ndimage.rotate(img, angle[2], (1, 2), reshape=False)
                lbl = ndimage.rotate(lbl, angle[0], (1, 2), reshape=False, order=0)
                lbl = ndimage.rotate(lbl, angle[1], (0, 2), reshape=False, order=0)
                lbl = ndimage.rotate(lbl, angle[2], (0, 1), reshape=False, order=0)
            else:
                angle = (torch.rand(3) - 0.5) * 60
                img = ndimage.rotate(img, angle[2], (1, 2), reshape=False)
                lbl = ndimage.rotate(lbl, angle[2], (0, 1), reshape=False, order=0)

        return img, lbl


class RandomGaussianBlur:
    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, sample):
        img, lbl = sample
        if torch.rand(1) > self.prob:
            # Each dimension is blurred with a probability of 50%
            mask = torch.randint(2, (img.ndim-1,))
            stddev = torch.rand(3) + 0.5
            for axis in range(img.ndim-1):
                if mask[axis] == 1:
                    img = ndimage.gaussian_filter1d(img, float(stddev[axis]), axis=axis+1)

        return img, lbl


class RandomGaussianNoise:
    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, sample):
        img, lbl = sample
        if torch.rand(1) > self.prob:
            stddev = torch.rand(1) + 0.1
            img += np.array(torch.randn(img.shape) * stddev)
        return img, lbl


class RandomBrightnessAdjustment:
    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, sample):
        img, lbl = sample
        if torch.rand(1) <= self.prob:
            img *= float((torch.rand(1) * 0.6 + 0.7))

        return img, lbl
