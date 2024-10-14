import math
import torch
from torch import nn
from torch.nn import functional as F

from typing import List
import todos

import pdb

def in_projection_packed(q, k, v, w, b) -> List[torch.Tensor]:
    # w.size() -- [1536, 512]
    # b.size() -- [1536]
    w_q, w_k, w_v = w.chunk(3)
    # (Pdb) w_q.size() -- [512, 512]
    # (Pdb) w_k.size() -- [512, 512]
    # (Pdb) w_v.size() -- [512, 512]

    b_q, b_k, b_v = b.chunk(3)
    # (Pdb) b_q.size(), b_k.size(), b_v.size() -- [512], [512], [512]
    k = k.to(q.dtype) # for half ?

    # torch.nn.functional.linear(input, weight, bias=None) → Tensor
    # y = F.linear(v, w_v, b_v)
    # tensor [v] size: [1024, 1, 256], min: -11.04214, max: 12.239643, mean: 0.041581
    # tensor [w_v] size: [256, 256], min: -0.072583, max: 0.07293, mean: 3.4e-05
    # tensor [b_v] size: [256], min: -0.00482, max: 0.003588, mean: -7e-05
    # tensor [y] size: [1024, 1, 256], min: -6.726958, max: 7.565886, mean: -0.01203
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v) # ggml_debug


def multi_head_attention_forward(query, key, value, num_heads: int,
    in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias):

    # (Pdb) query.size() -- [100, 1, 256]
    # (Pdb) key.size() -- [1024, 1, 256]
    # (Pdb) value.size() -- [1024, 1, 256]

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape # torch.Size([100, 1, 256])

    head_dim = embed_dim // num_heads # 32 -- 256/8

    q, k, v = in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    # tensor [q] size: [100, 1, 256], min: -3.425014, max: 3.135417, mean: -0.006095
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1) # [100, 8, 32] ==> [8, 100, 32]
    # tensor [q] size: [8, 100, 32], min: -3.425014, max: 3.135417, mean: -0.006095

    # tensor [k] size: [1024, 1, 256], min: -7.79951, max: 7.048189, mean: -0.051877
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    # tensor [k] size: [8, 1024, 32], min: -7.79951, max: 7.048189, mean: -0.051877

    # tensor [v] size: [1024, 1, 256], min: -6.726958, max: 7.565886, mean: -0.01203
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    # tensor [v] size: [8, 1024, 32], min: -6.726958, max: 7.565886, mean: -0.01203

    B, Nt, E = q.shape # [8, 100, 32]
    q_scaled = q / math.sqrt(E)
    # k.size() -- [8, 1024, 32]
    # k.transpose(-2, -1).size() -- [8, 32, 1024]
    attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    # tensor [attn_output_weights] size: [8, 100, 1024], min: -6.5267, max: 8.720252, mean: 0.013913
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    # tensor [attn_output_weights] size: [8, 100, 1024], min: 0.0, max: 0.250939, mean: 0.000977

    attn_output = torch.bmm(attn_output_weights, v)
    # tensor [attn_output] size: [8, 100, 32], min: -3.122495, max: 3.007122, mean: -0.015471

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim) # [8, 100, 32] ==> [100, 8, 32] ==> [100, 256]

    # (Pdb) out_proj_weight.size() -- [256, 256], out_proj_bias.size() -- [256]
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1)) # [100, 1, 256]
    # tensor [attn_output] size: [100, 1, 256], min: -2.726733, max: 2.047928, mean: -0.011761

    return attn_output

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim # 256
        self.num_heads = num_heads # 8
        self.in_proj_weight = nn.Parameter(torch.zeros((3 * embed_dim, embed_dim)))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias) # bias == True

    def forward(self, query, key, value):
        # (Pdb) query.size() -- [100, 1, 256]
        # (Pdb) key.size() -- [1024, 1, 256]
        # (Pdb) value.size() -- [1024, 1, 256]
        attn_output = multi_head_attention_forward(
            query, key, value,
            self.num_heads,
            self.in_proj_weight, 
            self.in_proj_bias,
            self.out_proj.weight, 
            self.out_proj.bias,
        )
        return attn_output # [100, 1, 256]

    def extra_repr(self) -> str:
        return f"embed_dim = {self.embed_dim}, num_heads = {self.num_heads}"


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        # nn.MultiheadAttention has error on ONNX export, so we replace it
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        # self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, query_pos):
        return tensor + query_pos

    def forward(self, tgt, query_pos):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)
        # tgt = tgt + self.dropout(tgt2)
        tgt = tgt + tgt2
        tgt = self.norm(tgt)

        return tgt

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        # nn.MultiheadAttention has error on ONNX export, so we replace it
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        # self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor + pos

    def forward(self, tgt, memory, pos, query_pos):
        # tensor [tgt] size: [100, 1, 256], min: -3.530838, max: 3.635915, mean: -0.000685
        # tensor [memory] size: [1024, 1, 256], min: -11.04214, max: 12.239643, mean: 0.041581
        # tensor [pos] size: [1024, 1, 256], min: -1.0, max: 1.0, mean: 0.494228
        # tensor [query_pos] size: [100, 1, 256], min: -3.593544, max: 3.98427, mean: 0.010172
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)
        # tensor [tgt2] size: [100, 1, 256], min: -2.726733, max: 2.047928, mean: -0.011761
        # tgt = tgt + self.dropout(tgt2)
        tgt = tgt + tgt2
        tgt = self.norm(tgt)
        
        return tgt


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor + pos

    def forward(self, tgt):
        # tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        # tgt = tgt + self.dropout(tgt2)
        tgt2 = self.linear2(F.relu(self.linear1(tgt)))
        tgt = tgt + tgt2

        tgt = self.norm(tgt)
        return tgt

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        # input_dim = 256
        # hidden_dim = 256
        # output_dim = 256
        # num_layers = 3
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        # self = MLP(
        #   (layers): ModuleList(
        #     (0-2): 3 x Linear(in_features=256, out_features=256, bias=True)
        #   )
        # )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            x = layer(x)
            if i < self.num_layers - 1:
                x = F.relu(x)

        return x
