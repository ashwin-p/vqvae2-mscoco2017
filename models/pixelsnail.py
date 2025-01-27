# models/pixelsnail.py
# Copyright (c) Xi Chen
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Borrowed from https://github.com/neocxi/pixelsnail-public and ported it to PyTorch
import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt
from functools import partial, lru_cache
import numpy as np


def wn_linear(in_dim, out_dim):
    return nn.utils.parametrizations.weight_norm(nn.Linear(in_dim, out_dim))


class WNConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, activation=None):
        super().__init__()
        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias)
        )
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def shift_down(x, size=1):
    return F.pad(x, [0, 0, size, 0])[:, :, : x.shape[2], :]


def shift_right(x, size=1):
    return F.pad(x, [size, 0, 0, 0])[:, :, :, : x.shape[3]]


class CausalConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding='downright', activation=None):
        super().__init__()
        self.conv = WNConv2d(in_channel, out_channel, kernel_size, padding=0, activation=activation)
        if padding == 'downright':
            self.pad = nn.ZeroPad2d([kernel_size - 1, 0, kernel_size - 1, 0])
        else:
            self.pad = nn.ZeroPad2d([kernel_size // 2, kernel_size // 2, kernel_size - 1, 0])

    def forward(self, x):
        return self.conv(self.pad(x))


class PixelBlock(nn.Module):
    def __init__(self, in_channel, channel, kernel_size, n_res_block, attention=True, dropout=0.1, condition_dim=0):
        super().__init__()
        self.blocks = nn.ModuleList([WNConv2d(in_channel, channel, kernel_size, padding=kernel_size // 2)] +
                                    [WNConv2d(channel, channel, kernel_size, padding=kernel_size // 2) for _ in range(n_res_block)])
        self.attention = attention
        self.condition_dim = condition_dim
        if condition_dim > 0:
            self.cond_resnet = WNConv2d(condition_dim, channel, 1)
        self.out = WNConv2d(channel, in_channel, 1)

    def forward(self, x, condition=None):
        for block in self.blocks:
            x = block(x)
        if self.condition_dim > 0 and condition is not None:
            x += self.cond_resnet(condition)
        return self.out(x)


class PixelSNAIL(nn.Module):
    def __init__(self, shape, n_class, channel, kernel_size, n_block, n_res_block, res_channel, attention=True,
                 dropout=0.1, condition_dim=0):
        super().__init__()
        height, width = shape
        self.horizontal = CausalConv2d(n_class, channel, kernel_size, padding='down')
        self.vertical = CausalConv2d(n_class, channel, kernel_size, padding='downright')
        self.blocks = nn.ModuleList([PixelBlock(channel, res_channel, kernel_size, n_res_block, attention, dropout,
                                                condition_dim=condition_dim) for _ in range(n_block)])
        self.out = WNConv2d(channel, n_class, 1)

    def forward(self, x, condition=None):
        x = self.horizontal(x) + self.vertical(x)
        for block in self.blocks:
            x = block(x, condition)
        return self.out(x)

