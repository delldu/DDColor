import torch
import torch.nn as nn
from torch.nn import functional as F
import todos
import pdb


def custom_conv_layer(ni, nf,
    ks = 3,
    stride = 1,
    use_activ = True, # True | False
    extra_bn = False, # True || False
):
    padding = (ks - 1) // 2
    conv = nn.Conv2d(ni, nf, kernel_size=ks, bias=not extra_bn, stride=stride, padding=padding)
    conv = nn.utils.spectral_norm(conv)
    layers = [conv]

    if use_activ: # True | False
        layers.append(nn.ReLU(True))
    if extra_bn: # True | False
        layers.append(nn.BatchNorm2d(nf))

    return nn.Sequential(*layers)

# ggml_debug
class CustomPixelShuffle(nn.Module):
    def __init__(self, ni, nf,
                 scale=2, # 2 | 4
                 extra_bn=False): # True | False
        super().__init__()
        # ni = 1536
        # nf = 512
        # scale = 2

        self.conv = custom_conv_layer(ni, nf * (scale**2), ks=1, use_activ=False, extra_bn=extra_bn)
        # self.conv ----
        # Sequential(
        #   (0): Conv2d(1536, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #   (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # )
        self.shuf = nn.PixelShuffle(scale)
        
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x))


class UnetBlockWide(nn.Module):
    def __init__(self, up_in_c, x_in_c, n_out):
        super().__init__()
        # up_in_c = 1536
        # x_in_c = 768
        # n_out = 512
        self.shuf = CustomPixelShuffle(up_in_c, n_out, scale=2, extra_bn=True)
        # (Pdb) self.shuf
        # CustomPixelShuffle(
        #   (conv): Sequential(
        #     (0): Conv2d(1536, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   )
        #   (shuf): PixelShuffle(upscale_factor=2)
        #   (pad): ReplicationPad2d((1, 0, 1, 0))
        #   (blur): AvgPool2d(kernel_size=2, stride=1, padding=0)
        #   (relu): ReLU(inplace=True)
        # )
        self.bn = nn.BatchNorm2d(x_in_c) # ggml_debug

        ni = n_out + x_in_c # 1280
        self.conv = custom_conv_layer(ni, n_out, ks =3, extra_bn=True)
        # (Pdb) self.conv
        # Sequential(
        #   (0): Conv2d(1280, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        #   (1): ReLU(inplace=True)
        #   (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # )
        self.relu = nn.ReLU()


    def forward(self, up_in, s):
        # tensor [up_in] size: [1, 1536, 16, 16], min: -26.544884, max: 16.26297, mean: -0.00155
        # tensor [s] size: [1, 768, 32, 32], min: -6.887635, max: 24.325577, mean: 0.001135
        up_out = self.shuf(up_in)
        # tensor [up_out] size: [1, 512, 32, 32], min: 0.0, max: 4.052812, mean: 0.314582

        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        # tensor [cat_x] size: [1, 1280, 32, 32], min: 0.0, max: 8.217813, mean: 0.350186
        # tensor [self.conv(cat_x)] size: [1, 512, 32, 32], min: -2.069555, max: 33.400291, mean: -0.

        # --------------------------------------------------------------------------------------------
        # ggml_debug
        # self.bn -- BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        # self.bn.state_dict() is dict:
        # tensor [weight] size: [768], min: 0.780468, max: 1.366361, mean: 0.912081
        # tensor [bias] size: [768], min: -0.136402, max: 0.466912, mean: 0.102002
        # tensor [running_mean] size: [768], min: -0.962075, max: 16.190048, mean: 0.001125
        # tensor [running_var] size: [768], min: 0.0012, max: 51.826649, mean: 0.180806
        # tensor [num_batches_tracked] size: [], min: 310000.0, max: 310000.0, mean: 310000.0
        # tensor [s] size: [1, 768, 32, 32], min: -6.887635, max: 24.325577, mean: 0.001135
        # tensor [self.bn(s)] size: [1, 768, 32, 32], min: -8.827082, max: 8.217813, mean: 0.092955
        # --------------------------------------------------------------------------------------------
        return self.conv(cat_x)
