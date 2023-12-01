import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import acb
from model.acb_old import ACBlock
from model.diversebranchblock import DiverseBranchBlock

def default_conv(in_channels, out_channels, kernel_size, bias=True, deploy=False, norm="batch"):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def default_acb(in_channels, out_channels, kernel_size, stride=1, padding=-1, dilation=1, groups=1, 
                bias=True, padding_mode='zeros', use_original_conv=False, deploy=False, norm="batch"):
    if padding == -1:
        padding = (kernel_size//2)
    if use_original_conv or kernel_size == 1 or kernel_size == (1, 1) or kernel_size >= 7:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                        dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    else:
        return acb.ACBlock(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                        dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=deploy, norm=norm)

def default_dbb(in_channels, out_channels, kernel_size, stride=1, padding=-1, dilation=1, groups=1, 
                bias=True, padding_mode='zeros', use_original_conv=False, deploy=False, norm="batch"):
    if padding == -1:
        padding = (kernel_size//2)
    return DiverseBranchBlock(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                        dilation=dilation, groups=groups, deploy=deploy)

def default_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=-1, dilation=1, groups=1, 
                bias=True, padding_mode='zeros', use_original_conv=False, deploy=False, norm="batch"):
    if padding == -1:
        padding = (kernel_size//2)
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                        dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

def default_acbv5(in_channels, out_channels, kernel_size, stride=1, padding=-1, dilation=1, groups=1, 
                bias=True, padding_mode='zeros', use_original_conv=False, deploy=False, norm="batch"):
    if padding == -1:
        padding = (kernel_size//2)
    if use_original_conv or kernel_size == 1 or kernel_size == (1, 1) or kernel_size >= 7:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                        dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    else:
        return ACBlock(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                        dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=deploy, norm=norm)

# 因为EDSR框架在读取时已经将RGB转成0-255的值了，因此不需要像swinir那样做转化，仅添加bias偏移即可实现mean
class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=False, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True, deploy=False, norm="batch"):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias=bias, deploy=deploy, norm=norm))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
                elif act == 'gelu':
                    m.append(nn.GELU())

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias=bias, deploy=deploy, norm=norm))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
            elif act == 'gelu':
                m.append(nn.GELU())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class UpsamplerDirect(nn.Sequential):
    def __init__(self, conv, scale, n_feats, n_out_ch, bias=True, deploy=False, norm="batch"):

        m = []
        m.append(conv(n_feats, (scale ** 2) * n_out_ch, 3, bias=bias, deploy=deploy, norm=norm))
        m.append(nn.PixelShuffle(scale))
        super(UpsamplerDirect, self).__init__(*m)

class Downsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.PixelUnshuffle(2))
                m.append(conv(n_feats * 4, n_feats, 3, bias))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(nn.PixelUnshuffle(3))
            m.append(conv(n_feats * 9, n_feats, 3, bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Downsampler, self).__init__(*m)

