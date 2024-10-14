# Copyright (c) Facebook, Inc. and its affiliates.
# Modified from: https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
import todos
import pdb

class PositionEmbedding(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        # num_pos_feats = 128
        # temperature = 10000

        self.scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32) # [0.0, 127.0]
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats) # 1, 1, 1, 128
        self.dim_t = nn.Parameter(dim_t.reshape(1, 1, 1, num_pos_feats), requires_grad=False)
        # self.register_buffer('dim_t', dim_t.reshape(1, 1, 1, num_pos_feats))

        # struct ggml_tensor * ggml_arange(
        #     struct ggml_context * ctx,
        #     float start,
        #     float stop,
        #     float step)

    def forward(self, x):
        # x.size():  [1, 512, 32, 32] | [1, 512, 64, 64] | [1, 256, 128, 128]
        B, C, H, W = x.size()
        # not_mask = torch.ones((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) # [1, 32, 32]
        not_mask = torch.ones((B, H, W), device=x.device, dtype=torch.bool) # [1, 32, 32]
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = y_embed.transpose(1, 2) # not_mask.cumsum(2, dtype=torch.float32)

        # (Pdb) y_embed.size() -- [1, 32, 32]
        # tensor([[[ 1.,  1.,  1.,  ...,  1.,  1.,  1.],
        #          [ 2.,  2.,  2.,  ...,  2.,  2.,  2.],
        #          [ 3.,  3.,  3.,  ...,  3.,  3.,  3.],
        #          ...,
        #          [30., 30., 30.,  ..., 30., 30., 30.],
        #          [31., 31., 31.,  ..., 31., 31., 31.],
        #          [32., 32., 32.,  ..., 32., 32., 32.]]], device='cuda:0')

        # normalize:
        eps = 1e-6
        # y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        # x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        y_embed = y_embed / (H + eps) * self.scale
        x_embed = x_embed / (W + eps) * self.scale

        # self.dim_t.size() -- [128]
        pos_x = x_embed[:, :, :, None] / self.dim_t.to(x.device) # [1, 32, 32, 1] --> [1, 32, 32, 128]
        pos_y = y_embed[:, :, :, None] / self.dim_t.to(x.device)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # ------------------------------------------------------------------------------------------------
        # tensor [pos_x] size: [1, 32, 32, 128], min: -1.0, max: 1.0, mean: 0.494228
        # tensor [pos_y] size: [1, 32, 32, 128], min: -1.0, max: 1.0, mean: 0.494228
        # tensor [torch.cat((pos_y, pos_x), dim=3)] size: [1, 32, 32, 256], min: -1.0, max: 1.0, mean: 0.494228
        # tensor [pos] size: [1, 256, 32, 32], min: -1.0, max: 1.0, mean: 0.494228
        # ------------------------------------------------------------------------------------------------

        # tensor [pos_x] size: [1, 128, 128, 128], min: -1.0, max: 1.0, mean: 0.494896
        # tensor [pos_y] size: [1, 128, 128, 128], min: -1.0, max: 1.0, mean: 0.494896
        # tensor [cat(pos_y, pos_x)] size: [1, 128, 128, 256], min: -1.0, max: 1.0, mean: 0.494896
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # tensor [pos] size: [1, 256, 128, 128], min: -1.0, max: 1.0, mean: 0.494896

        return pos
