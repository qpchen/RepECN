import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# from utils.modules.lightWeightNet import WeightNet
from . import lapar_weightnet as LW

def make_model(args, parent=False):
    return LAPAR_B(args)

class ComponentDecConv(nn.Module):
    def __init__(self, k_path, k_size):
        super(ComponentDecConv, self).__init__()

        kernel = pickle.load(open(k_path, 'rb'))
        kernel = torch.from_numpy(kernel).float().view(-1, 1, k_size, k_size)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        out = F.conv2d(x, weight=self.weight, bias=None, stride=1, padding=0, groups=1)

        return out

# patch size default 64
class LAPAR_B(nn.Module):
    # def __init__(self, config):
    def __init__(self, args):
        super(LAPAR_B, self).__init__()

        self.k_size = 5#config.MODEL.KERNEL_SIZE
        self.s = args.scale[0]#config.MODEL.SCALE
        in_chl = args.n_colors
        nf = 24
        n_block = 3
        out_chl = 72
        scale = args.scale[0]

        # self.w_conv = WeightNet(config.MODEL)
        # self.decom_conv = ComponentDecConv(config.MODEL.KERNEL_PATH, self.k_size)
        self.w_conv = LW.WeightNet(in_chl, nf, n_block, out_chl, scale)
        self.decom_conv = ComponentDecConv('./model/lapara_kernel_72_k5.pkl', self.k_size)

        self.criterion = nn.L1Loss(reduction='mean')


    def forward(self, x, gt=None):
        B, C, H, W = x.size()

        bic = F.interpolate(x, scale_factor=self.s, mode='bicubic', align_corners=False)
        pad = self.k_size // 2
        x_pad = F.pad(bic, pad=(pad, pad, pad, pad), mode='reflect')
        pad_H, pad_W = x_pad.size()[2:]
        x_pad = x_pad.view(B * 3, 1, pad_H, pad_W)
        x_com = self.decom_conv(x_pad).view(B, 3, -1, self.s * H, self.s * W)  # B, 3, N_K, Hs, Ws

        weight = self.w_conv(x)
        weight = weight.view(B, 1, -1, self.s * H, self.s * W)  # B, 1, N_K, Hs, Ws

        out = torch.sum(weight * x_com, dim=2)

        if gt is not None:
            loss_dict = dict(L1=self.criterion(out, gt))
            return loss_dict
        else:
            return out
