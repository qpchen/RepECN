import torch
from torch import nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return IDN(args)

class FBlock(nn.Module):
    def __init__(self, num_features):
        super(FBlock, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05)
        )

    def forward(self, x):
        return self.module(x)


class DBlock(nn.Module):
    def __init__(self, num_features, d, s):
        super(DBlock, self).__init__()
        self.num_features = num_features
        self.s = s
        self.enhancement_top = nn.Sequential(
            nn.Conv2d(num_features, num_features - d, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(num_features - d, num_features - 2 * d, kernel_size=3, padding=1, groups=4),
            nn.LeakyReLU(0.05),
            nn.Conv2d(num_features - 2 * d, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05)
        )
        self.enhancement_bottom = nn.Sequential(
            nn.Conv2d(num_features - d, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(num_features, num_features - d, kernel_size=3, padding=1, groups=4),
            nn.LeakyReLU(0.05),
            nn.Conv2d(num_features - d, num_features + d, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05)
        )
        self.compression = nn.Conv2d(num_features + d, num_features, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.enhancement_top(x)
        slice_1 = x[:, :int((self.num_features - self.num_features/self.s)), :, :]
        slice_2 = x[:, int((self.num_features - self.num_features/self.s)):, :, :]
        x = self.enhancement_bottom(slice_1)
        x = x + torch.cat((residual, slice_2), 1)
        x = self.compression(x)
        return x


class IDN(nn.Module):
    def __init__(self, args):
        super(IDN, self).__init__()
        self.scale = args.scale[0]
        # num_features = args.n_feats
        # d = args.d
        # s = args.s
        num_features = 64
        d = 16
        s = 4

        self.fblock = FBlock(num_features)
        self.dblocks = nn.Sequential(*[DBlock(num_features, d, s) for _ in range(4)])
        self.deconv = nn.ConvTranspose2d(num_features, 3, kernel_size=17, stride=self.scale, padding=8, output_padding=self.scale-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        bicubic = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        x = self.fblock(x)
        x = self.dblocks(x)
        x = self.deconv(x)
        return bicubic + x