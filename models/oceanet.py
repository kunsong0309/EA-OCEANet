"""
OCEANet 

Orthogonal Cascaded Epileptiform Attention Network

A unified attention architecture for cross-species epileptiform activity detection

Vita Shaw
"""

import torch
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer
from einops.layers.torch import Rearrange, Reduce


class ResidualConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        drop_rate=0.5,
        norm_group=4,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
                bias=False,
            ),
            nn.GroupNorm(norm_group, out_channels),
            nn.ELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
                bias=False,
            ),
            nn.GroupNorm(norm_group, out_channels),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            nn.GroupNorm(norm_group, out_channels),
        )

        self.add = nn.Sequential(
            nn.ELU(),
            nn.MaxPool2d(
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.add(self.conv1(x) + self.conv2(x))


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        drop_rate=0.5,
        norm_group=4,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
                bias=False,
            ),
            nn.GroupNorm(norm_group, out_channels),
            nn.ELU(),
            nn.MaxPool2d(
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.conv(x)


class PatchEmbedding(nn.Module):
    def __init__(self, emb_type='dense', num_features=32, kernel_size=7, n_blocks=2,
                 norm_group=4, drop_rate=0., downsample=4,
                 ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(
                1,
                num_features,
                (1, kernel_size),
                stride=(1, 2),
                padding=(0, (kernel_size - 1) // 2),
                bias=False,
            ),
            nn.GroupNorm(norm_group, num_features),
            nn.ELU(),
            nn.MaxPool2d((1, 3), (1, 2), (0, 1)),
        )

        if emb_type == 'residual':
            self.projection.append(nn.Sequential(
                *[ResidualConvBlock(
                    num_features * (2 ** i),
                    num_features * (2 ** i) * 2,
                    norm_group=norm_group,
                    drop_rate=drop_rate,
                ) for i in range(n_blocks)],
                nn.AvgPool2d((1, downsample), (1, downsample)),
            ))

        elif emb_type == 'shallow':
            self.projection.append(nn.Sequential(
                *[ConvBlock(
                    num_features * (2 ** i),
                    num_features * (2 ** i) * 2,
                    norm_group=norm_group,
                    drop_rate=drop_rate,
                ) for i in range(n_blocks)],
                nn.AvgPool2d((1, downsample), (1, downsample)),
            ))

        else:
            NotImplemented

    def forward(self, x):
        # batch_size, 1, num_chan * num_patch, seg_len = x.shape
        return self.projection(x)


class PositionEncoding(nn.Module):
    def __init__(
        self,
        emb_size=128,
        num_heads=8,
        depth=4,
        drop_rate=0.5,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            Rearrange('b e (c) (w) -> b (c w) e'),
            LinearAttentionTransformer(
                dim=emb_size,
                heads=num_heads,
                depth=depth,
                max_seq_len=1024,
                attn_layer_dropout=drop_rate,
                attn_dropout=drop_rate,
            ),
        )

    def forward(self, x):
        return self.encoder(x).mean(1)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        emb_size=128,
        depth=4,
        num_heads=8,
        drop_rate=0.5,
    ):
        super().__init__()

        self.msa_chan = LinearAttentionTransformer(
            dim=emb_size,
            heads=num_heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=drop_rate,
            attn_dropout=drop_rate,
        )

        self.msa_ts = LinearAttentionTransformer(
            dim=emb_size,
            heads=num_heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=drop_rate,
            attn_dropout=drop_rate,
        )

    def forward(self, x):
        batch_size, emb_size, num_chan, num_patch = x.shape
        x = x.transpose(1, 3).contiguous().view(
            batch_size*num_patch, num_chan, emb_size)
        x = self.msa_chan(x).mean(1)
        x = x.contiguous().view(
            batch_size, num_patch, emb_size)
        x = self.msa_ts(x).mean(1)
        return x


class OCEANet(nn.Module):
    def __init__(self,
                 in_channels=16, n_classes=2, n_samples=2000,
                 patch_length=200, emb_type='residual',
                 num_features=64, n_blocks=6, norm_group=8,
                 kernel_size=7, encode_type='msa',
                 emb_size=128, num_heads=8, depth=6,
                 drop_rate=0.5, drop_cnn=0., use_pretrained=None,
                 freeze_embedding=False, freeze_encoder=False):
        super(OCEANet, self).__init__()

        dfac = (2 ** (n_blocks + 2))
        width = patch_length // dfac
        num_patch = n_samples // dfac // width

        self.embedding = PatchEmbedding(emb_type=emb_type,
                                        num_features=num_features,
                                        kernel_size=kernel_size,
                                        n_blocks=n_blocks,
                                        norm_group=norm_group,
                                        downsample=width,
                                        drop_rate=drop_cnn,
                                        )

        if encode_type == 'msa':
            self.encoder = PositionEncoding(emb_size=emb_size,
                                            num_heads=num_heads,
                                            depth=depth,
                                            drop_rate=drop_rate,
                                            )
        elif encode_type == 'msata':
            self.encoder = TransformerEncoder(emb_size=emb_size,
                                              num_heads=num_heads,
                                              depth=depth,
                                              drop_rate=drop_rate,
                                              )
        elif encode_type == 'mlp':
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(emb_size * in_channels * num_patch,
                          emb_size * num_patch),
                nn.ELU(),
                nn.Dropout(0.5),
                nn.Linear(num_patch * emb_size, emb_size),
            )
        else:
            self.encoder = nn.Sequential(
                Reduce('b e c w -> b e', reduction='mean'),
            )

        self.classifier = nn.Sequential(
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(emb_size, n_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.bias.data.zero_()

        if use_pretrained is not None:
            match_module = ['embedding', 'encoder']
            # match_module = ['embedding']
            pretrained_dict = torch.load(use_pretrained, map_location='cuda')
            pretrained_dict = {n: m for n, m in pretrained_dict.items()
                               if n.split('.')[0] in match_module}
            model_dict = self.state_dict()
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

        if freeze_embedding:
            self.embedding.requires_grad_(False)
        if freeze_encoder:
            self.encoder.requires_grad_(False)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        pool = Reduce('b e n -> b e', reduction='mean')
        return pool(x)
