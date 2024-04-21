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

    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

def multi_head_attention_forward(query, key, value, num_heads: int,
    in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias):

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    head_dim = embed_dim // num_heads

    q, k, v = in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1) # size() -- [8, 256, 64]

    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    B, Nt, E = q.shape
    q_scaled = q / math.sqrt(E)
    attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)

    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    return attn_output

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.in_proj_weight = nn.Parameter(torch.zeros((3 * embed_dim, embed_dim)))

        self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)


    def forward(self, query, key, value):
        attn_output = multi_head_attention_forward(
            query, key, value, 
            self.num_heads,
            self.in_proj_weight, 
            self.in_proj_bias,
            self.out_proj.weight, 
            self.out_proj.bias,
        )
        return attn_output


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
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)

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
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
