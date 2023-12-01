from torch import nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return SRCNN(args)


class SRCNN(nn.Module):
    def __init__(self, args):
        super(SRCNN, self).__init__()
        num_channels = args.n_colors
        self.scale = args.scale[0]
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
        fea = self.relu(self.conv1(x))
        fea = self.relu(self.conv2(fea))
        out = self.conv3(fea)
        return out
