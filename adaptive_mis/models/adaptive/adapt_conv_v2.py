import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from torch.nn.parameter import Parameter
import math
import scipy as sp
import scipy.linalg as linalg
import numpy as np
import pdb
import sys

try:
    from .fb import *
except:
    from fb import *
from torch.nn.utils import spectral_norm


class AdaptiveConv2(nn.Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'num_bases',
                     'bases_grad', 'mode']

    def __init__(self, in_channels, inter=0, adaptive_kernel_max_size=3, adaptive_kernel_min_size=3, inter_kernel_size=3, stride=1, padding=0,
                 num_bases=6, bias=True, bases_grad=True, dilation=1, groups=1,
                 mode='mode1', bases_drop=None, drop_rate=0.0, **kwargs):
        super().__init__()
        self.drop_rate = drop_rate
        self.adaptive_kernel_max_size = adaptive_kernel_max_size
        self.in_channels = in_channels
        self.inter_kernel_size = inter_kernel_size
        self.features = 6
        self.kernel_size = adaptive_kernel_max_size
        self.stride = stride
        self.padding = adaptive_kernel_max_size // 2
        self.num_bases = num_bases
        assert mode in ['mode0', 'mode1'], 'Only mode0 and mode1 are available at this moment.'
        self.mode = mode
        self.bases_grad = bases_grad
        self.dilation = dilation
        self.bases_drop = bases_drop
        self.groups = groups

        bases = bases_list(adaptive_kernel_min_size, adaptive_kernel_max_size, num_bases)
        self.register_buffer('bases', torch.Tensor(bases).float())
        self.tem_size = len(bases) // num_bases
        # print("bbbbb", self.tem_size)
        # bases_size = num_bases * len(bases)
        bases_size = len(bases)

        inter = inter or in_channels

        self.new_out_channels = self.features * bases_size
        # print("out", self.new_out_channels)
        self.out_channels = self.new_out_channels
        self.bases_net = nn.Sequential(nn.Conv2d(in_channels, inter, kernel_size=inter_kernel_size, padding=inter_kernel_size // 2, stride=stride),
                                       nn.BatchNorm2d(inter),
                                       nn.Tanh(),)
        for i in range(1, len(bases)):
            self.bases_net.append(nn.Sequential(
                nn.Conv2d(inter, inter, kernel_size=inter_kernel_size, padding=inter_kernel_size // 2, stride=stride),
                nn.BatchNorm2d(inter),
                nn.Tanh(),
            ))

        self.bases_net.append(nn.Sequential(
            nn.Conv2d(inter, self.out_channels, kernel_size=inter_kernel_size, padding=inter_kernel_size // 2),
            nn.BatchNorm2d(self.out_channels),
            nn.Tanh()
        )
        )
        # print(self.bases_net)
        self.coef = Parameter(torch.Tensor(self.out_channels, in_channels * num_bases, 1, 1))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.drop = Bases_Drop(p=0.1)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.coef.size(1))

        nn.init.kaiming_normal_(self.coef, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        N, C, H, W = input.shape

        M = self.num_bases

        drop_rate = self.drop_rate
        bases = self.bases_net(F.dropout2d(input, p=drop_rate, training=self.training)).view(
            N, self.features, len(self.bases), H // self.stride, W // self.stride)  # BxMxMxHxW

        # self.bases_coef = bases.cpu().data.numpy()
        bases = torch.einsum('bmkhw, kl->bmlhw', bases, self.bases)
        # self.bases_save = bases.cpu().data.numpy()

        x = F.unfold(F.dropout2d(input, p=drop_rate, training=self.training),
                     kernel_size=self.adaptive_kernel_max_size, stride=self.stride, padding=self.padding)
        x = x.view(N, C, self.kernel_size * self.kernel_size, H // self.stride, W // self.stride)
        bases_out = torch.einsum('bmlhw, bclhw-> bcmhw',
                                 bases.view(N, self.num_bases, -1, H // self.stride, W // self.stride),
                                 x).reshape(N, self.in_channels * self.num_bases, H // self.stride, W // self.stride)
        bases_out = F.dropout2d(bases_out, p=drop_rate, training=self.training)

        #out = F.conv2d(bases_out, self.coef, self.bias)
        out = bases_out

        return out

    def extra_repr(self):
        return 'kernel_size={kernel_size}, inter_kernel_size={inter_kernel_size}, stride={stride}, padding={padding}, num_bases={num_bases}' \
            ', bases_grad={bases_grad}, mode={mode}, bases_drop={bases_drop}, in_channels={in_channels}, out_channels={out_channels}'.format(**self.__dict__)


class Bases_Drop(nn.Module):
    def __init__(self, p):
        super(Bases_Drop, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            assert len(x.shape) == 5
            N, M, L, H, W = x.shape
            mask = torch.ones((N, 1, L, H, W)).float().cuda() * (1 - self.p)
            mask = torch.bernoulli(mask) * (1 / (1 - self.p))
            x = x * mask
        return x


def bases_list(adaptive_kernel_min_size, adaptive_kernel_max_size, num_bases):

    b_list = []
    for kernel_size in range(adaptive_kernel_min_size, adaptive_kernel_max_size + 1, 2):
        i = kernel_size // 2 - 1
        normed_bases, _, _ = calculate_FB_bases(i + 1)
        normed_bases = normed_bases.transpose().reshape(-1, kernel_size, kernel_size).astype(np.float32)[:num_bases, ...]

        pad = adaptive_kernel_max_size // 2 - (i + 1)
        bases = torch.Tensor(normed_bases)
        # print(i, kernel_size, bases.shape, normed_bases.shape, pad, num_bases, adaptive_kernel_max_size)
        bases = F.pad(bases, (pad, pad, pad, pad, 0, 0)).view(num_bases, adaptive_kernel_max_size * adaptive_kernel_max_size)
        b_list.append(bases)
    return torch.cat(b_list, 0)


if __name__ == '__main__':
    layer = AdaptiveConv2(3, inter=3, inter_kernel_size=3, padding=1, stride=1, bias=True,
                          adaptive_kernel_max_size=9, adaptive_kernel_min_size=3)  # .cuda()
    print(layer.out_channels)

    normal_layer = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=9, padding=1, stride=1, bias=True,),
        # nn.Conv2d(layer.out_channels, layer.out_channels, kernel_size=7, padding=1, stride=1, bias=True,),
        # nn.Conv2d(3, layer.out_channels, kernel_size=5, padding=1, stride=1, bias=True,),
        # nn.Conv2d(3, layer.out_channels, kernel_size=3, padding=1, stride=1, bias=True,),
        # nn.Conv2d(1, 1, kernel_size=7, padding=1, stride=1, bias=True,),
        # nn.Conv2d(1, 1, kernel_size=5, padding=1, stride=1, bias=True,),
        # nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=True,)
    )
    data = torch.randn(10, 3, 224, 224)  # .cuda()
    print(sum(p.numel() for p in layer.parameters() if p.requires_grad))
    print(sum(p.numel() for p in normal_layer .parameters() if p.requires_grad))
    import time
    start = time.time()
    print(layer(data).shape)
    print("time", time.time() - start)
    start = time.time()
    print(normal_layer(data).shape)
    print("normal_time", time.time() - start)


model = nn.Sequential(
    nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=True,),
    nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=True,),
    # nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=True,),
    nn.Conv2d(1, 12, kernel_size=3, padding=1, stride=1, bias=True,))
sum(p.numel() for p in model.parameters() if p.requires_grad)
