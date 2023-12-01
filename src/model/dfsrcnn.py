from math import sqrt
from model import common

import torch
from torch import nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return DFSRCNN(args)

class DFSRCNN(nn.Module):
    """
    Args:
        upscale_factor (int): Image magnification factor.
    """

    # def __init__(self, upscale_factor: int) -> None:
    def __init__(self, args):
        super(DFSRCNN, self).__init__()

        upscale_factor = args.scale[0]
        num_channels = args.n_colors

        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        ####################################################
        # Feature extraction layer.
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, 56, (5, 5), (1, 1), (2, 2)),
            nn.PReLU(56)
        )

        # Shrinking layer.
        self.shrink = nn.Sequential(
            nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(12)
        )

        # Mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12)
        )

        # Expanding layer.
        self.expand = nn.Sequential(
            nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(56)
        )

        # Deconvolution layer.
        self.deconv = nn.ConvTranspose2d(56, num_channels, (9, 9), (upscale_factor, upscale_factor),
                                         (4, 4), (upscale_factor - 1, upscale_factor - 1))

        #################################################
        # Bi-direct training process layers
        # Bi-direct Deconv layer
        self.bideconv = nn.Sequential(
            nn.Conv2d(num_channels, 56, (9, 9), (upscale_factor, upscale_factor), (4, 4)),
            nn.PReLU(56)
        )

        # Bi-direct Expanding layer
        self.biexpand = nn.Sequential(
            nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(12)
        )

        # Bi-direct Mapping layer.
        self.bimap = nn.Sequential(
            self.map[6],
            nn.PReLU(12),
            self.map[4],
            nn.PReLU(12),
            self.map[2],
            nn.PReLU(12),
            self.map[0],
            nn.PReLU(12)
        )

        # Bi-direct Shrinking layer
        self.bishrink = nn.Sequential(
            nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(56)
        )

        # Bi-direct Feature extraction layer
        self.bifeature = nn.Conv2d(56, num_channels, (5, 5), (1, 1), (2, 2))

        #################################################
        # Initialize model weights.
        self._initialize_weights()

    def forward(self, y, isfeaOut = False):
        ############################
        # training from HR to LR
        bifea1 = self.bideconv(y)
        bifea2 = self.biexpand(bifea1)
        bifea3 = self.bimap(bifea2)
        bifea4 = self.bishrink(bifea3)
        biout = self.bifeature(bifea4)

        if isfeaOut:
            return biout, bifea1, bifea2, bifea3, bifea4
        else:
            return biout

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0).
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias.data)
