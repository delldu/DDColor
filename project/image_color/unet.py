from enum import Enum
import torch
import torch.nn as nn
from torch.nn import functional as F
import collections
import todos
import pdb

NormType = Enum('NormType', 'Batch BatchZero Weight Spectral')


class Hook:
    feature = None

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self.feature = output
        elif isinstance(output, collections.OrderedDict):
            self.feature = output['out']

    def remove(self):
        self.hook.remove()


class SelfAttention(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.query = conv1d(n_channels, n_channels // 8)
        self.key = conv1d(n_channels, n_channels // 8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        #Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


def batchnorm_2d(nf: int, norm_type: NormType = NormType.Batch):
    "A batchnorm2d layer with `nf` features initialized depending on `norm_type`."
    bn = nn.BatchNorm2d(nf)
    with torch.no_grad():
        bn.bias.fill_(1e-3)
        bn.weight.fill_(0. if norm_type == NormType.BatchZero else 1.)
    return bn


def init_default(m: nn.Module, func=nn.init.kaiming_normal_) -> None:
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func:
        if hasattr(m, 'weight'): func(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)
    return m


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function."
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale**2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return nn.utils.spectral_norm(conv)


def custom_conv_layer(
    ni: int,
    nf: int,
    ks: int = 3,
    stride: int = 1,
    use_activ: bool = True, # True | False
    extra_bn: bool = False, # True || False
):
    padding = (ks - 1) // 2
    conv_func = nn.Conv2d
    conv = init_default(
        conv_func(ni, nf, kernel_size=ks, bias=not extra_bn, stride=stride, padding=padding),
        nn.init.kaiming_normal_,
    )
    conv = nn.utils.spectral_norm(conv)
    layers = [conv]

    if use_activ: # True | False
        layers.append(nn.ReLU(True))
    if extra_bn: # True | False
        layers.append(nn.BatchNorm2d(nf))

    return nn.Sequential(*layers)


class CustomPixelShuffle_ICNR(nn.Module):
    def __init__(self,
                 ni: int,
                 nf: int = None,
                 scale: int = 2, # 2 | 4
                 extra_bn=False): # True | False
        super().__init__()
        # ni = 1536
        # nf = 512
        # scale = 2

        self.conv = custom_conv_layer(
            ni, nf * (scale**2), ks=1, use_activ=False, extra_bn=extra_bn)
        # self.conv ----
        # Sequential(
        #   (0): Conv2d(1536, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #   (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # )
        icnr(self.conv[0].weight)
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
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."

    def __init__(self,
                 up_in_c: int,
                 x_in_c: int,
                 n_out: int,
                 hook,
            ):
        super().__init__()

        self.hook = hook
        up_out = n_out
        self.shuf = CustomPixelShuffle_ICNR(up_in_c, up_out, extra_bn=True)
        self.bn = batchnorm_2d(x_in_c)
        ni = up_out + x_in_c
        self.conv = custom_conv_layer(ni, n_out, extra_bn=True)

        self.relu = nn.ReLU()

    def forward(self, up_in):
        s = self.hook.feature
        print("s ===self.hook.feature==== ", s.size())

        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv(cat_x)
