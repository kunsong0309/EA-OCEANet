"""
Jing, Jin, Wendong Ge, Shenda Hong, Marta Bento Fernandes, Zhen Lin, Chaoqi Yang, Sungtae An et al. "Development of expert-level classification of seizures
    and rhythmic and periodic patterns during EEG interpretation." Neurology 100, no. 17 (2023): e1750-e1762.

https://github.com/bdsp-core/IIIC-SPaRCNet adapted by Vita Shaw
"""
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Sequential):
    def __init__(
        self,
        in_channels,
        growth_rate,
        bn_size,
        drop_rate,
        conv_bias,
        batch_norm,
    ):
        super(_DenseLayer, self).__init__()
        if batch_norm:
            self.add_module("norm1", nn.BatchNorm1d(in_channels))
        self.add_module("elu1", nn.ELU())
        self.add_module(
            "conv1",
            nn.Conv1d(
                in_channels,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=conv_bias,
            )
        )
        if batch_norm:
            self.add_module("norm2", nn.BatchNorm1d(bn_size * growth_rate))
        self.add_module("elu2", nn.ELU())
        self.add_module(
            "conv2",
            nn.Conv1d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=conv_bias,
            )
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features,
                p=self.drop_rate,
                training=self.training
            )
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self,
        num_layers,
        in_channels,
        growth_rate,
        bn_size,
        drop_rate,
        conv_bias,
        batch_norm,
    ):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
                conv_bias,
                batch_norm,
            )
            self.add_module("denselayer%d" % (i + 1), layer)


class _TransitionLayer(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_bias,
        batch_norm,
    ):
        super(_TransitionLayer, self).__init__()
        if batch_norm:
            self.add_module("norm", nn.BatchNorm1d(in_channels))
        self.add_module("elu", nn.ELU())
        self.add_module(
            "conv",
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=conv_bias,
            ),
        )
        self.add_module("pool", nn.AvgPool1d(kernel_size=2, stride=2))


class SPaRCNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        out_channels_init=32,
        n_blocks=4,
        block_layers=4,
        growth_rate=16,
        bn_size=16,
        drop_rate=0.5,
        conv_bias=True,
        batch_norm=True,
        **kwargs,
    ):
        super(SPaRCNet, self).__init__()

        # add initial convolutional layer
        first_conv = OrderedDict(
            [
                (
                    "conv0",
                    nn.Conv1d(
                        in_channels,
                        out_channels_init,
                        kernel_size=7,
                        stride=2,
                        padding=3,
                        bias=conv_bias,
                    ),
                )
            ]
        )
        if batch_norm:
            first_conv["norm0"] = nn.BatchNorm1d(out_channels_init)
        first_conv["elu0"] = nn.ELU()
        first_conv["pool0"] = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.densenet = nn.Sequential(first_conv)

        n_channels = out_channels_init

        # add dense blocks
        for i in np.arange(n_blocks):
            block = _DenseBlock(
                num_layers=block_layers,
                in_channels=n_channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
            )
            self.densenet.add_module(f"denseblock{(i + 1):d}", block)
            # update number of channels after each dense block
            n_channels = n_channels + block_layers * growth_rate

            if i != (n_blocks - 1):
                trans = _TransitionLayer(
                    in_channels=n_channels,
                    out_channels=n_channels // 2,
                    conv_bias=conv_bias,
                    batch_norm=batch_norm,
                )
                self.densenet.add_module(f"transition{(i + 1):d}", trans)
                # update number of channels after each transition layer
                n_channels = n_channels // 2

        if batch_norm:
            self.densenet.add_module(
                f"norm{(n_blocks + 1):d}", nn.BatchNorm1d(n_channels))
        self.densenet.add_module(f"relu{(n_blocks + 1):d}", nn.ReLU())
        self.densenet.add_module(
            f"pool{(n_blocks + 1):d}", nn.AdaptiveAvgPool1d(1))

        # add classifier
        self.num_features = n_channels
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_channels, n_classes),
        )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = x.squeeze(1)
        features = self.densenet(x).squeeze(-1)
        out = self.classifier(features)
        return out

