import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import CustomPixelShuffle, UnetBlockWide, custom_conv_layer

from .convnext import ConvNeXt
from .transformer_utils import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from .position_encoding import PositionEmbeddingSine

from typing import List, Tuple

import todos
import pdb
from ggml_engine import create_network

# [norm0, norm1, norm2, norm3]
ENCODER_RESULT = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

class DDColor(nn.Module):
    def __init__(self,
                 num_input_channels=3,
                 nf=512,
                 num_output_channels=2,
                 num_queries=100,
                 num_scales=3,
                 dec_layers=9,
                ):
        super().__init__()
        self.MAX_H = 512 # Fixed, DO NOT change !!!
        self.MAX_W = 512 # Fixed, DO NOT change !!!
        self.MAX_TIMES = 1
        # GPU - 3G, 100ms, CPU -- 1300ms

        self.encoder = Encoder('convnext-l')

        self.decoder = Decoder(
            nf=nf,
            num_queries=num_queries,
            num_scales=num_scales,
            dec_layers=dec_layers,
        )
        self.refine_net = nn.Sequential(
            custom_conv_layer(num_queries + 3, num_output_channels, ks=1, use_activ=False))
        # (Pdb) self.refine_net
        # Sequential(
        #   (0): Sequential(
        #     (0): Conv2d(103, 2, kernel_size=(1, 1), stride=(1, 1))
        #   )
        # )

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.load_weights()

        # Remove spectral normal 
        nn.utils.remove_spectral_norm(self.refine_net[0][0])
        for i in range(3):
            nn.utils.remove_spectral_norm(self.decoder.layers[i].conv[0])
            nn.utils.remove_spectral_norm(self.decoder.layers[i].shuf.conv[0])
        nn.utils.remove_spectral_norm(self.decoder.last_shuf.conv[0])

    def load_weights(self, model_path="models/image_color.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        self.load_state_dict(torch.load(checkpoint))

    def normalize(self, img):
        return (img - self.mean) / self.std # ggml_debug

    def forward(self, x):
        assert x.size(2) == 512 and x.size(3) == 512, "Please input 1x3x512x512 tensor"
        x = self.normalize(x)
        # tensor [x] size: [1, 3, 512, 512], min: -2.117904, max: 2.326308, mean: -0.356781
        
        # How Avoiding Pitfalls in onnx exporting ??? !!!
        encoder_layers: ENCODER_RESULT = self.encoder(x)
        # encoder_layers is tuple: len = 4
        #     tensor [item] size: [1, 192, 128, 128], min: -10.815851, max: 5.158478, mean: -0.007829
        #     tensor [item] size: [1, 384, 64, 64], min: -13.253959, max: 16.581171, mean: -0.000148
        #     tensor [item] size: [1, 768, 32, 32], min: -6.887635, max: 24.325577, mean: 0.001135
        #     tensor [item] size: [1, 1536, 16, 16], min: -26.544884, max: 16.26297, mean: -0.00155

        out_feat = self.decoder(encoder_layers)
        coarse_input = torch.cat([out_feat, x], dim=1)
        out_ab = self.refine_net(coarse_input)
        # tensor [out_feat] size: [1, 100, 512, 512], min: -14.656635, max: 24.051313, mean: 1.814844
        # tensor [coarse_input] size: [1, 103, 512, 512], min: -14.656635, max: 24.051313, mean: 1.751593
        # tensor [out_ab] size: [1, 2, 512, 512], min: -41.829132, max: 52.045349, mean: 4.471401

        return out_ab


class Decoder(nn.Module):
    def __init__(self, 
                nf=512,
                num_queries=100,
                num_scales=3,
                dec_layers=9,
            ):
        super().__init__()
        self.layers = self.make_layers()
        embed_dim = nf // 2 # --> 256

        self.last_shuf = CustomPixelShuffle(embed_dim, embed_dim, scale=4)
        # (Pdb) self.last_shuf
        # CustomPixelShuffle(
        #   (conv): Sequential(
        #     (0): Conv2d(256, 4096, kernel_size=(1, 1), stride=(1, 1))
        #   )
        #   (shuf): PixelShuffle(upscale_factor=4)
        #   (pad): ReplicationPad2d((1, 0, 1, 0))
        #   (blur): AvgPool2d(kernel_size=2, stride=1, padding=0)
        #   (relu): ReLU(inplace=True)
        # )
        self.color_decoder = MultiScaleColorDecoder(
            in_channels=[512, 512, 256],
            num_queries=num_queries, # 100
            num_scales=num_scales, # 3
            dec_layers=dec_layers, # 9
        )

    def make_layers(self):
        # nn.Sequential only accept one tensor as input
        # under torch.jit.script, so we replace with ModuleList !!!

        # decoder_layers = []
        # decoder_layers.append(UnetBlockWide(1536, 768, 512))
        # decoder_layers.append(UnetBlockWide(512, 384, 512))
        # decoder_layers.append(UnetBlockWide(512, 192, 256))
        # return nn.Sequential(*decoder_layers)

        m = nn.ModuleList()
        m.append(UnetBlockWide(1536, 768, 512))
        m.append(UnetBlockWide(512, 384, 512))
        m.append(UnetBlockWide(512, 192, 256))
        return m
        

    def layer_output(self, layer_index: int, x, s_x):
        '''Ugly code for support torch.jit.script'''
        for i, layer in enumerate(self.layers):
            if i == layer_index:
                x = layer(x, s_x)
        return x


    def forward(self, encoder_output_layers: ENCODER_RESULT):
        # encoder_output_layers is tuple: len = 4
        #     tensor [item] size: [1, 192, 128, 128], min: -10.815851, max: 5.158478, mean: -0.007829
        #     tensor [item] size: [1, 384, 64, 64], min: -13.253959, max: 16.581171, mean: -0.000148
        #     tensor [item] size: [1, 768, 32, 32], min: -6.887635, max: 24.325577, mean: 0.001135
        #     tensor [item] size: [1, 1536, 16, 16], min: -26.544884, max: 16.26297, mean: -0.00155

        (x0, x1, x2, x3) = encoder_output_layers
        out0 = self.layer_output(0, x3, x2)
        # tensor [out0] size: [1, 512, 32, 32], min: -2.069555, max: 33.400291, mean: -0.006212

        out1 = self.layer_output(1, out0, x1)
        # tensor [out1] size: [1, 512, 64, 64], min: -2.505532, max: 12.87324, mean: -0.048077

        out2 = self.layer_output(2, out1, x0)
        # tensor [out2] size: [1, 256, 128, 128], min: -2.547425, max: 12.310425, mean: -0.095538

        out3 = self.last_shuf(out2) 
        # tensor [out3] size: [1, 256, 512, 512], min: 0.0, max: 0.285963, mean: 0.009585

        out = self.color_decoder([out0, out1, out2], out3)
        # tensor [out] size: [1, 100, 512, 512], min: -14.656635, max: 24.051313, mean: 1.814844

        return out


class Encoder(nn.Module):
    def __init__(self, encoder_name):
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

    def forward(self, x) -> ENCODER_RESULT:
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
                    d_model=hidden_dim, # 256
                    nhead=nheads, # 8
                    dropout=0.0,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim, # 256
                    nhead=nheads, # 8
                    dropout=0.0,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim, # 256
                    dim_feedforward=dim_feedforward, # 2048
                    dropout=0.0,
                )
            )
        
        self.decoder_norm = nn.LayerNorm(hidden_dim) # hidden_dim === 256
        # (Pdb) self.decoder_norm -- LayerNorm((256,), eps=1e-05, elementwise_affine=True)

        # learnable color query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # self.query_feat -- Embedding(100, 256)

        # learnable color query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding
        self.num_feature_levels = num_scales # 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # input projections
        self.input_proj = nn.ModuleList()
        for i in range(self.num_feature_levels): # 3
            self.input_proj.append(nn.Conv2d(in_channels[i], hidden_dim, kernel_size=1))
        # (Pdb) self.input_proj
        # ModuleList(
        #   (0-1): 2 x Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        #   (2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        # )

        # output FFNs
        self.color_embed = MLP(hidden_dim, hidden_dim, color_embed_dim, 3) # input_dim, hidden_dim, output_dim, num_layers
        # (Pdb) self.color_embed
        # MLP(
        #   (layers): ModuleList(
        #     (0-2): 3 x Linear(in_features=256, out_features=256, bias=True)
        #   )
        # )

    def forward(self, x: List[torch.Tensor], img_features):
        # x is list: len = 3
        #     tensor [item] size: [1, 512, 32, 32], min: -2.036234, max: 162340.96875, mean: 3001.375
        #     tensor [item] size: [1, 512, 64, 64], min: -2.632989, max: 835685120.0, mean: 68049544.0
        #     tensor [item] size: [1, 256, 128, 128], min: -2.94301, max: 1553683709952.0, mean: 143206252544.0
        # tensor [img_features] size: [1, 256, 512, 512], min: 0.0, max: 222161108992.0, mean: 1800433664.0

        src = []
        pos = []
        # for i in range(self.num_feature_levels): # 3
        #     pos_temp = self.pe_layer(x[i]).flatten(2).permute(2, 0, 1)
        #     # self.level_embed.weight.size() -- [3, 256]
        #     src_temp = self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None]
        #     src_temp = src_temp.permute(2, 0, 1)
        #     pos.append(pos_temp)
        #     src.append(src_temp)
        for i, layer in enumerate(self.input_proj):
            # x[0].size() -- [1, 512, 32, 32]
            # self.pe_layer(x[0]).size() -- [1, 256, 32, 32]
            # self.pe_layer(x[0]).flatten(2).size() -- [1, 256, 1024]
            pos_temp = self.pe_layer(x[i]).flatten(2).permute(2, 0, 1) # [1024, 1, 256]

            # self.level_embed.weight.size() -- [3, 256]
            src_temp = layer(x[i]).flatten(2) + self.level_embed.weight[i][None, :, None]
            src_temp = src_temp.permute(2, 0, 1)

            pos.append(pos_temp)
            src.append(src_temp)

        bs = src[0].shape[1] # src[0].shape -- [1024, 1, 256]

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1) # [100, 1024, 256]
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1) # [100, 1024, 256]
        # for i in range(self.dec_layers): # 9
        #     level_index = i % self.num_feature_levels # 3
        #     # attention: cross-attention first
        #     output = self.transformer_cross_attention_layers[i](
        #         output, src[level_index],
        #         pos=pos[level_index], query_pos=query_embed
        #     )
        #     output = self.transformer_self_attention_layers[i](output, query_pos=query_embed)
        #     # FFN
        #     output = self.transformer_ffn_layers[i](output)
        i = 0
        for (cross_layer, self_layer, ffn_layer) in zip(
            self.transformer_cross_attention_layers, 
            self.transformer_self_attention_layers,
            self.transformer_ffn_layers):

            level_index = i % self.num_feature_levels # 3
            # attention: cross-attention first
            output = cross_layer(output, src[level_index], pos=pos[level_index], query_pos=query_embed)
            output = self_layer(output, query_pos=query_embed)
            # FFN
            output = ffn_layer(output)
            i = i + 1

        # output.size() -- [100, 1, 256]
        decoder_output = self.decoder_norm(output) # size() -- [100, 1, 256]
        decoder_output = decoder_output.transpose(0, 1)  # size() -- [1, 100, 256]
        color_embed = self.color_embed(decoder_output)
        out = torch.einsum("bqc,bchw->bqhw", color_embed, img_features) # ggml_debug

        # tensor [color_embed] size: [1, 100, 256], min: -32.289177, max: 50.403393, mean: 0.213315
        # tensor [img_features] size: [1, 256, 512, 512], min: 0.0, max: 0.285963, mean: 0.009585
        # tensor [out] size: [1, 100, 512, 512], min: -14.656635, max: 24.051313, mean: 1.814844

        return out # size() -- [1, 100, 512, 512]
