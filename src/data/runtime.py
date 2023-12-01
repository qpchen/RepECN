import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

class Runtime(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Runtime, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'runtime', self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubicL')
        else:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('', '.png')

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        i_dim = lr.ndim
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        ########################################################
        # Add by CQP: if model output only Y channel, convert to RGB version.
        if self.args.n_colors == 1 and i_dim == 3:
            lr_cbcr = common.rgb2cbcr(lr)
            lr_cbcr_t = common.np2Tensor(*lr_cbcr, rgb_range=self.args.rgb_range)
        elif self.args.n_colors == 1 and i_dim == 2:
            lr_cbcr = [lr] * 2
            lr_cbcr_t = [torch.from_numpy(i).float() for i in lr_cbcr]
        
        ########################################################

        if self.args.n_colors == 1 and not self.train:
            return pair_t[0], pair_t[1], lr_cbcr_t[0], lr_cbcr_t[1], i_dim, filename
        else:
            return pair_t[0], pair_t[1], filename
