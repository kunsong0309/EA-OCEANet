"""
Lawhern, Vernon J., et al. "EEGNet: a compact convolutional neural network for EEG-based brain–computer 
    interfaces." Journal of neural engineering 15.5 (2018): 056013.

https://github.com/vlawhern/arl-eegmodels adapted by Vita Shaw
"""

import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class EEGNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        n_samples=200,
        F1=8,
        F2=16,
        D=2,
        kernel_length=64,
        drop_rate=0.5,
        norm_rate=0.25,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            Conv2dWithConstraint(F1, D * F1, (in_channels, 1),
                                 bias=False, groups=F1, max_norm=1.),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(drop_rate),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, (1, 16), padding='same',
                      bias=False, groups=D * F1),
            nn.Conv2d(D * F1, F2, (1, 1), padding='same', bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 8)),
            nn.Dropout(drop_rate),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(n_samples // 32 * F2,
                                 n_classes, max_norm=norm_rate),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x
