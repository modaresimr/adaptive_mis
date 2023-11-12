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


class AdaptiveConv3(nn.Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'num_bases',
                     'bases_grad', 'mode']

    def __init__(self, in_channels,features=6, inter=0, adaptive_kernel_max_size=3, adaptive_kernel_min_size=3, inter_kernel_size=3, stride=1, padding=0,
                 num_bases=6, bias=True, bases_grad=True, dilation=1, groups=1,
                 mode='mode1', bases_drop=None, drop_rate=0.0, **kwargs):
        super().__init__()
        self.drop_rate = drop_rate
        self.adaptive_kernel_max_size = adaptive_kernel_max_size
        self.in_channels = in_channels
        self.inter_kernel_size = inter_kernel_size
        self.features = features
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

        self.generator_out_channels= self.features * bases_size
        # print("out", self.new_out_channels)
        self.out_channels = self.in_channels * self.num_bases
        
        self.bases_net = nn.Sequential(nn.Conv2d(in_channels, inter, kernel_size=inter_kernel_size, padding=inter_kernel_size // 2, stride=stride),
                                       nn.BatchNorm2d(inter),
                                       nn.Tanh(),
                                       )
        for i in range(1, len(bases)):
            self.bases_net.append(nn.Sequential(
                nn.Conv2d(inter, inter, kernel_size=inter_kernel_size, padding=inter_kernel_size // 2, stride=stride),
                # nn.BatchNorm2d(inter),
                # nn.Tanh(),
            ))

        self.bases_net.append(nn.Sequential(
            nn.Conv2d(inter, self.generator_out_channels, kernel_size=inter_kernel_size, padding=inter_kernel_size // 2),
            nn.BatchNorm2d(self.generator_out_channels),
            nn.Tanh()
        )
        )
        # print(self.bases_net)
        self.coef = Parameter(torch.Tensor(self.generator_out_channels, in_channels * num_bases, 1, 1))

        if bias:
            self.bias = Parameter(torch.Tensor(self.generator_out_channels))
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
        drop=F.dropout2d(input, p=drop_rate, training=self.training)
        bases = self.bases_net(drop)
        bases=bases.view(N, self.features, len(self.bases), H // self.stride, W // self.stride)  # BxMxMxHxW
        # return bases
        # self.bases_coef = bases.cpu().data.numpy()
        # print("bases", bases.shape, self.bases.shape)
        # # bases = torch.einsum('bmkhw, kl->bmlhw', bases, self.bases)
        # # bases = torch.matmul(bases, self.bases)
        # # Reshape A to collapse the last two dimensions into one
        # A=bases
        # B=self.bases

        # # Assuming A and B are defined with the following shapes
        # # A.shape = [10, 6, 24, 224, 224]
        # # B.shape = [24, 81]

        # # Permute and reshape A to bring the '24' dimension to the last and flatten the others
        # A_permuted = A.permute(0, 1, 3, 4, 2)  # New shape: [10, 6, 224, 224, 24]
        # A_reshaped = A_permuted.reshape(-1, len(self.bases))  # New shape: [10 * 6 * 224 * 224, 24]

        # # Perform matrix multiplication with B
        # C_intermediate = torch.matmul(A_reshaped, B)  # New shape: [10 * 6 * 224 * 224, 81]

        # # Reshape the result back to a 5D tensor and permute to get the desired shape
        # bases = C_intermediate.reshape(N, M, W, H, self.bases.shape[1]).permute(0, 1, 4, 2, 3)  # Final shape: [10, 6, 81, 224, 224]
        bases=optimize_einsum1(bases,self.bases)

        
        # self.bases_save = bases.cpu().data.numpy()

        x = F.unfold(drop,kernel_size=self.adaptive_kernel_max_size, stride=self.stride, padding=self.padding)
        x = x.view(N, C, self.kernel_size * self.kernel_size, H // self.stride, W // self.stride)
        newKernel=bases.view(N, self.num_bases, -1, H // self.stride, W // self.stride)
        # print(newKernel.shape, x.shape)
        # bases_out = torch.einsum('bmlhw, bclhw-> bcmhw',newKernel,x)
        bases_out=optimize_einsum2(newKernel,x)
        kernel_out=bases_out.reshape(N, self.in_channels * self.num_bases, H // self.stride, W // self.stride)
        kernel_out = F.dropout2d(kernel_out, p=drop_rate, training=self.training)

        #out = F.conv2d(bases_out, self.coef, self.bias)
        out = kernel_out

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


def optimize_einsum1(A,B):
    #replacment of torch.einsum('bmkhw, kl->bmlhw',A,B)
    # return torch.einsum('bmkhw, kl->bmlhw',A,B)
    N, F, BASSES, H , W =A.shape  # BxMxMxHxW
    K,D=B.shape
    assert K==BASSES
    # print("A=",A.shape," B=",B.shape)
    A_permuted = A.permute(0, 1, 3, 4, 2)  # New shape: [10, 6, 224, 224, 24]
    A_reshaped = A_permuted.reshape(-1, K)  # New shape: [10 * 6 * 224 * 224, 24]

    # Perform matrix multiplication with B
    C_intermediate = torch.matmul(A_reshaped, B)  # New shape: [10 * 6 * 224 * 224, 81]

    # Reshape the result back to a 5D tensor and permute to get the desired shape
    bases = C_intermediate.view(N,F, H, W, D).permute(0, 1, 4, 2, 3)  # Final shape: [10, 6, 81, 224, 224]
    return bases

def optimize_einsum2(A,B):
    # optimized version of bases_out = torch.einsum('bmlhw, bclhw-> bcmhw',A,B)
    # return torch.einsum('bmlhw, bclhw-> bcmhw',A,B)
    N, M, L, H, W = A.shape
    _,C,_,_,_=B.shape
    A=A.permute(1,0,3,4,2).reshape(M,-1,L)
    B=B.permute(1,0,3,4,2).reshape(C,-1,L)
    # print("A=",A.shape," B=",B.shape)
    bases_out = torch.einsum('mol, col-> cmo', A, B)
    bases_out=bases_out.view(C, M, N, H, W).permute(2,0,1,3,4)
    return bases_out

from auto_profiler import Profiler




if __name__ == '__main__':
  with Profiler():
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




