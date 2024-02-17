import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import Hook, CustomPixelShuffle_ICNR,  UnetBlockWide, NormType, custom_conv_layer
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
                 last_norm='Spectral',
                 num_queries=100,
                 num_scales=3,
                 dec_layers=9,
                ):
        super().__init__()
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 1

        self.encoder = Encoder(encoder_name, ['norm0', 'norm1', 'norm2', 'norm3'])
        self.encoder.eval()
        test_input = torch.randn(1, num_input_channels, *input_size)
        self.encoder(test_input)

        self.decoder = Decoder(
            self.encoder.hooks,
            nf=nf,
            last_norm=last_norm,
            num_queries=num_queries,
            num_scales=num_scales,
            dec_layers=dec_layers,
        )
        self.refine_net = nn.Sequential(custom_conv_layer(num_queries + 3, num_output_channels, ks=1, 
            use_activ=False, norm_type=NormType.Spectral))
    
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
        out_feat = self.decoder()
        coarse_input = torch.cat([out_feat, x], dim=1)
        out = self.refine_net(coarse_input)

        return out.float()


class Decoder(nn.Module):
    def __init__(self, hooks,
                 nf=512,
                 blur=True,
                 last_norm='Spectral',
                 num_queries=100,
                 num_scales=3,
                 dec_layers=9,
                ):
        super().__init__()
        # hooks ?????
        # [<basicsr.archs.ddcolor_arch_utils.unet.Hook object at 0x7fe9d5cfa640>, 
        # <basicsr.archs.ddcolor_arch_utils.unet.Hook object at 0x7fe9cc1c94f0>, 
        # <basicsr.archs.ddcolor_arch_utils.unet.Hook object at 0x7fe9cc1c95b0>, 
        # <basicsr.archs.ddcolor_arch_utils.unet.Hook object at 0x7fe9cc1c9670>]

        self.hooks = hooks
        self.nf = nf
        self.blur = blur
        self.last_norm = getattr(NormType, last_norm)

        self.layers = self.make_layers()
        embed_dim = nf // 2

        self.last_shuf = CustomPixelShuffle_ICNR(embed_dim, embed_dim, blur=self.blur, 
            norm_type=self.last_norm, scale=4)
        
        self.color_decoder = MultiScaleColorDecoder(
            in_channels=[512, 512, 256],
            num_queries=num_queries,
            num_scales=num_scales,
            dec_layers=dec_layers,
        )

    def forward(self):
        # self.hooks[-1] -- <basicsr.archs.ddcolor_arch_utils.unet.Hook object>
        encode_feat = self.hooks[-1].feature # size() -- [1, 1536, 16, 16] ???
        out0 = self.layers[0](encode_feat)
        out1 = self.layers[1](out0) 
        out2 = self.layers[2](out1) 
        out3 = self.last_shuf(out2) 

        out = self.color_decoder([out0, out1, out2], out3)
           
        return out

    def make_layers(self):
        decoder_layers = []

        e_in_c = self.hooks[-1].feature.shape[1]
        in_c = e_in_c

        out_c = self.nf
        setup_hooks = self.hooks[-2::-1]
        # (Pdb) for k in self.hooks: print(k)
        # <image_color.unet.Hook object at 0x7f204c5141f0> # setup_hook3
        # <image_color.unet.Hook object at 0x7f1faac16e20> # setup_hook2
        # <image_color.unet.Hook object at 0x7f1faac16c10> # setup_hook1
        # <image_color.unet.Hook object at 0x7f1faac16f70>
        # ----------------------------
        # (Pdb) for k in setup_hooks: print(k)
        # <image_color.unet.Hook object at 0x7f1faac16c10>
        # <image_color.unet.Hook object at 0x7f1faac16e20>
        # <image_color.unet.Hook object at 0x7f204c5141f0>

        for layer_index, hook in enumerate(setup_hooks):
            feature_c = hook.feature.shape[1]
            if layer_index == len(setup_hooks) - 1:
                out_c = out_c // 2
            decoder_layers.append(UnetBlockWide(in_c, feature_c, out_c, hook, blur=self.blur, self_attention=False,
                    norm_type=NormType.Spectral))
            in_c = out_c
        return nn.Sequential(*decoder_layers)


class Encoder(nn.Module):

    def __init__(self, encoder_name, hook_names, **kwargs):
        super().__init__()
 
        if encoder_name == 'convnext-t' or encoder_name == 'convnext':
            self.arch = ConvNeXt()
        elif encoder_name == 'convnext-s':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
        elif encoder_name == 'convnext-b':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        elif encoder_name == 'convnext-l':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
        else:
            raise NotImplementedError

        # self.encoder_name = encoder_name
        self.hook_names = hook_names # ['norm0', 'norm1', 'norm2', 'norm3']
        self.hooks = self.setup_hooks()


    def setup_hooks(self):
        # self.hook_names -- ['norm0', 'norm1', 'norm2', 'norm3']
        # self.arch._modules['norm0'] -- LayerNormChannelsFirst
        # self.arch._modules['norm1'] -- LayerNormChannelsFirst
        # self.arch._modules['norm2'] -- LayerNormChannelsFirst
        # self.arch._modules['norm3'] -- LayerNormChannelsFirst
        
        hooks = [Hook(self.arch._modules[name]) for name in self.hook_names]
        return hooks

    def forward(self, x) -> List[torch.Tensor]:
        return self.arch(x)
    

class MultiScaleColorDecoder(nn.Module):
    def __init__(self, in_channels,
        hidden_dim=256,
        num_queries=100,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=9,
        pre_norm=False,
        color_embed_dim=256,
        enforce_input_project=True,
        num_scales=3
    ):
        super().__init__()

        # positional encoding
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
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
            if in_channels[i] != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels[i], hidden_dim, kernel_size=1))
                nn.init.kaiming_uniform_(self.input_proj[-1].weight, a=1)
                if self.input_proj[-1].bias is not None:
                    nn.init.constant_(self.input_proj[-1].bias, 0)
            else:
                self.input_proj.append(nn.Sequential())
                pdb.set_trace()

        # output FFNs
        self.color_embed = MLP(hidden_dim, hidden_dim, color_embed_dim, 3)

    def forward(self, x, img_features):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []

        for i in range(self.num_feature_levels):
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)    

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            # FFN
            output = self.transformer_ffn_layers[i](output)

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # [N, bs, C]  -> [bs, N, C]
        color_embed = self.color_embed(decoder_output)
        out = torch.einsum("bqc,bchw->bqhw", color_embed, img_features)

        return out
