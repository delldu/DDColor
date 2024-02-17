# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

import todos
import pdb

# [norm0, norm1, norm2, norm3]
ENCODER_RESULT = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNormChannelsLast(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=False)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = x + input
        return x

class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 27, 3], # [3, 3, 9, 3],
                 dims=[192, 384, 768, 1536], # [96, 192, 384, 768],
                 drop_path_rate=0.,
            ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNormChannelsFirst(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNormChannelsFirst(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # add norm layers for each output
        out_indices = (0, 1, 2, 3)
        for i in out_indices:
            layer = LayerNormChannelsFirst(dims[i], eps=1e-6)
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer

    def forward(self, x) -> ENCODER_RESULT:
        # tensor [x] size: [1, 3, 512, 512], min: -4.686537, max: 4.506181, mean: -0.000415

        # encoder_layers: List[torch.Tensor] = []
        x0 = x1 = x2 = x3 = x
        i = 0
        for (ds, st) in zip(self.downsample_layers, self.stages):
            x = ds(x)
            x = st(x)

            # self.norm0/1/2/3
            if i == 0:
                x0 = self.norm0(x)
            elif i == 1:
                x1 = self.norm1(x)
            elif i == 2:
                x2 = self.norm2(x)
            elif i == 3:
                x3 = self.norm3(x)
            i += 1

        return (x0, x1, x2, x3)

class LayerNormChannelsFirst(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class LayerNormChannelsLast(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
