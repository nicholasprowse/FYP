from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):
    """
    Convolutions with weight standardisation
    Apparently this smooths the loss landscape and improves training
    https://paperswithcode.com/method/weight-standardization
    """
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class StdConv3d(nn.Conv3d):
    """
    Convolutions with weight standardisation
    Apparently this smooths the loss landscape and improves training
    https://paperswithcode.com/method/weight-standardization
    """
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv3d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, dims=2, stride=1, groups=1, bias=False):
    conv = StdConv2d if dims == 2 else StdConv3d
    return conv(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, dims=2, stride=1, bias=False):
    conv = StdConv2d if dims == 2 else StdConv3d
    return conv(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, dims=2, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        # For ResNet50
        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, dims=dims, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, dims=dims, stride=stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, dims=dims, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # For ResNet34/18
        # self.gn1 = nn.GroupNorm(32, cout, eps=1e-6)
        # self.conv1 = conv3x3(cin, cout, dims=dims, stride=stride, bias=False)
        # self.gn2 = nn.GroupNorm(32, cout, eps=1e-6)
        # self.conv2 = conv3x3(cout, cout, dims=dims, bias=False)
        # self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, dims=dims, stride=stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # ResNet50
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        # ResNet18/34
        # y = self.relu(self.gn1(self.conv1(x)))
        # y = self.gn2(self.conv2(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    """
    Implementation of Pre-activation (v2) ResNet mode.
    Input has dimensions (B, 3, H, W) and experimentally it appears as though the output has dimensions
    (B, 1024*width_factor, H//16, W//16) where B is the batch size and the image size is HxW.
    Each feature has dimensions:
    [(B, 512*width_factor, H//8, W//8),
     (B, 256*width_factor, H//4, W//4),
     (B,  64*width_factor, H//2, W//2)]
    """

    def __init__(self, config):
        super().__init__()
        width = int(64 * config.resnet.width_factor)
        self.width = width
        self.dims = config.dims
        conv = StdConv2d if config.dims == 2 else StdConv3d
        self.root = nn.Sequential(OrderedDict([
            ('conv', conv(config.input_channels, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ])).float()

        pool = nn.MaxPool2d if config.dims == 2 else nn.MaxPool3d
        self.pool = pool(kernel_size=3, stride=2, padding=0)

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width, dims=config.dims))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width, dims=config.dims))
                    for i in range(2, config.resnet.num_layers[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2, dims=config.dims))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2, dims=config.dims))
                    for i in range(2, config.resnet.num_layers[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2, dims=config.dims))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4, dims=config.dims))
                    for i in range(2, config.resnet.num_layers[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = [x]
        x = self.root(x)
        features.append(x)
        x = self.pool(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            features.append(x)
        x = self.body[-1](x)
        return x, features[::-1]
