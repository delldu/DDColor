# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
import todos
import pdb

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

        x += input
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

    def forward(self, x) -> List[torch.Tensor]:
        # tensor [x] size: [1, 3, 512, 512], min: -4.686537, max: 4.506181, mean: -0.000415

        # for i in range(4):
        #     x = self.downsample_layers[i](x)
        #     x = self.stages[i](x)

        #     # add extra norm
        #     norm_layer = getattr(self, f'norm{i}')
        #     # self.norm0/1/2/3
        #     # x = norm_layer(x)
        #     norm_layer(x)

        output_layers: List[torch.Tensor] = []
        i = 0
        for (ds, st) in zip(self.downsample_layers, self.stages):
            x = ds(x)
            x = st(x)

            # self.norm0/1/2/3
            if i == 0:
                output_layers.append(self.norm0(x))
            elif i == 1:
                output_layers.append(self.norm1(x))
            elif i == 2:
                output_layers.append(self.norm2(x))
            elif i == 3:
                output_layers.append(self.norm3(x))
            i += 1

        # (Pdb) for i in range(len(output_layers)): print(output_layers[i].size())
        # torch.Size([1, 192, 128, 128])
        # torch.Size([1, 384, 64, 64])
        # torch.Size([1, 768, 32, 32])
        # torch.Size([1, 1536, 16, 16])
        return output_layers 

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
