import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .unet import Hook, CustomPixelShuffle_ICNR, UnetBlockWide, custom_conv_layer
from .unet import CustomPixelShuffle_ICNR, UnetBlockWide, custom_conv_layer

from .convnext import ConvNeXt
from .transformer_utils import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from .position_encoding import PositionEmbeddingSine

from typing import List

import todos
import pdb

class DDColor(nn.Module):
    def __init__(self,
                 encoder_name='convnext-l',
                 num_input_channels=3,
                 input_size=(512, 512),
                 nf=512,
                 num_output_channels=2,
                 num_queries=100,
                 num_scales=3,
                 dec_layers=9,
                ):
        super().__init__()
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 1

        self.encoder = Encoder(encoder_name, ['norm0', 'norm1', 'norm2', 'norm3'])

        self.decoder = Decoder(
            nf=nf,
            num_queries=num_queries,
            num_scales=num_scales,
            dec_layers=dec_layers,
        )
        self.refine_net = nn.Sequential(
            custom_conv_layer(num_queries + 3, num_output_channels, ks=1, use_activ=False))
    
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.load_weights()

    def load_weights(self, model_path="models/image_color.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        self.load_state_dict(torch.load(checkpoint))

    def normalize(self, img):
        return (img - self.mean) / self.std

    def forward(self, x):
        x = self.normalize(x)
        
        x = F.interpolate(x,
            size=(512, 512),
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False,
        )
        encoder_layers = self.encoder(x)
        out_feat = self.decoder(encoder_layers)
        coarse_input = torch.cat([out_feat, x], dim=1)
        out = self.refine_net(coarse_input)

        return out.float()


class Decoder(nn.Module):
    def __init__(self, 
                nf=512,
                num_queries=100,
                num_scales=3,
                dec_layers=9,
            ):
        super().__init__()
        self.nf = nf

        self.layers = self.make_layers()
        embed_dim = nf // 2

        self.last_shuf = CustomPixelShuffle_ICNR(embed_dim, embed_dim, scale=4)
        
        self.color_decoder = MultiScaleColorDecoder(
            in_channels=[512, 512, 256],
            num_queries=num_queries, # 100
            num_scales=num_scales, # 3
            dec_layers=dec_layers, # 9
        )

    def make_layers(self):
        decoder_layers = []
        decoder_layers.append(UnetBlockWide(1536, 768, 512))
        decoder_layers.append(UnetBlockWide(512, 384, 512))
        decoder_layers.append(UnetBlockWide(512, 192, 256))

        return nn.Sequential(*decoder_layers)
        

    def layer_output(self, layer_index: int, x, encoder_layers: List[torch.Tensor]):
        '''Ugly code for support torch.jit.script'''
        for i, layer in enumerate(self.layers):
            if i == layer_index:
                x = layer(x, encoder_layers)
        return x


    def forward(self, encoder_output_layers: List[torch.Tensor]):
        # print("-" * 120)
        # todos.debug.output_var("encoder_output_layers[0]", encoder_output_layers[0])
        # todos.debug.output_var("encoder_output_layers[1]", encoder_output_layers[1])
        # todos.debug.output_var("encoder_output_layers[2]", encoder_output_layers[2])
        # todos.debug.output_var("encoder_output_layers[3]", encoder_output_layers[3])
        # print("-" * 120)
        x = encoder_output_layers.pop()
        out0 = self.layer_output(0, x, encoder_output_layers)
        out1 = self.layer_output(1, out0, encoder_output_layers)
        out2 = self.layer_output(2, out1, encoder_output_layers)

        out3 = self.last_shuf(out2) 

        out = self.color_decoder([out0, out1, out2], out3)
        return out


class Encoder(nn.Module):

    def __init__(self, encoder_name, hook_names, **kwargs):
        super().__init__()
 
        if encoder_name == 'convnext-t' or encoder_name == 'convnext':
            self.arch = ConvNeXt()
        elif encoder_name == 'convnext-s':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
        elif encoder_name == 'convnext-b':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        elif encoder_name == 'convnext-l': # True
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
        else:
            raise NotImplementedError

        # self.hook_names = hook_names # ['norm0', 'norm1', 'norm2', 'norm3']

    def forward(self, x):
        return self.arch(x)
    

class MultiScaleColorDecoder(nn.Module):
    def __init__(self, in_channels,
        hidden_dim=256,
        num_queries=100,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=9,
        color_embed_dim=256,
        num_scales=3
    ):
        super().__init__()

        # positional encoding
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2)

        # define Transformer decoder here
        self.dec_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.dec_layers): # 9
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                )
            )
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # learnable color query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable color query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding
        self.num_feature_levels = num_scales
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # input projections
        self.input_proj = nn.ModuleList()
        for i in range(self.num_feature_levels):
            self.input_proj.append(nn.Conv2d(in_channels[i], hidden_dim, kernel_size=1))

        # output FFNs
        self.color_embed = MLP(hidden_dim, hidden_dim, color_embed_dim, 3)

    def forward(self, x, img_features):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels # 3
        src = []
        pos = []

        for i in range(self.num_feature_levels): # 3
            pos.append(self.pe_layer(x[i]).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)    

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(self.dec_layers): # 9
            level_index = i % self.num_feature_levels # 3
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                pos=pos[level_index], query_pos=query_embed
            )
            output = self.transformer_self_attention_layers[i](output,
                query_pos=query_embed
            )
            # FFN
            output = self.transformer_ffn_layers[i](output)

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # [N, bs, C]  -> [bs, N, C]
        color_embed = self.color_embed(decoder_output)
        out = torch.einsum("bqc,bchw->bqhw", color_embed, img_features)

        return out
