"""
Song, Yonghao and Zheng, Qingqing and Liu, Bingchuan and Gao, Xiaorong. "EEG conformer: Convolutional transformer for EEG decoding and visualization." 
    IEEE Transactions on Neural Systems and Rehabilitation Engineering 31 (2022): 710-719.

https://github.com/eeyhsong/EEG-Conformer adapted by Vita Shaw
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=22, out_channels=40, kernel_size=25,
                 emb_size=40, drop_rate=0.5, patch_len=75, hop_len=60):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, out_channels, (1, kernel_size), stride=(1, 1),
                      padding=(0, (kernel_size - 1) // 2)),
            nn.Conv2d(out_channels, out_channels,
                      (in_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.AvgPool2d((1, patch_len), (1, patch_len - hop_len)),
            nn.Dropout(drop_rate),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(out_channels, emb_size, (1, 1),
                      stride=(1, 1), bias=True),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(
            x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d",
                         h=self.num_heads)
        values = rearrange(self.values(
            x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, **kwargs):
        super().__init__(
            *[TransformerEncoderBlock(emb_size, **kwargs) for _ in range(depth)])


class EEGConformer(nn.Module):
    def __init__(self,
                 in_channels=22, n_samples=2000, n_classes=4, 
                 out_channels_init=40, kernel_size=25, emb_size=40, 
                 patch_length=75, hop_length=60, num_heads=10,
                 depth=6, drop_rate=0.5):
        super().__init__()
        self.embedding = PatchEmbedding(in_channels=in_channels,
                                        out_channels=out_channels_init,
                                        kernel_size=kernel_size,
                                        emb_size=emb_size,
                                        patch_len=patch_length,
                                        hop_len=hop_length,
                                        drop_rate=drop_rate,
                                        )
        self.encoder = TransformerEncoder(depth, emb_size, num_heads=num_heads,
                                          drop_p=drop_rate, forward_drop_p=drop_rate)
        
        n_patch = (n_samples - patch_length) // (patch_length - hop_length) + 1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_patch * emb_size, 256),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.classifier(x)
        return x
