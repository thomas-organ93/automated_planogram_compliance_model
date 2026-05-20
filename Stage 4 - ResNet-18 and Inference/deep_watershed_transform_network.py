import torch.nn as nn
from torch.nn import functional as F

class WatershedTransformNetwork(nn.Module):
    def __init__(self, in_channels=2, out_classes=16, internal_channels=64):
        super(WatershedTransformNetwork, self).__init__()

        # Block 1: 5x5 Conv -> AvgPool
        self.conv6 = nn.Conv2d(in_channels, internal_channels, kernel_size=5, padding=2)
        self.pool6 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Block 2: 5x5 Conv -> AvgPool
        self.conv7 = nn.Conv2d(internal_channels, internal_channels, kernel_size=5, padding=2)
        self.pool7 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Block 3: 1x1 Conv
        self.fcn7_1 = nn.Conv2d(internal_channels, internal_channels, kernel_size=1)
        self.fcn7_2 = nn.Conv2d(internal_channels, out_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]

        # Pass through blocks
        x = F.relu(self.conv6(x))
        x = self.pool6(x)

        x = F.relu(self.conv7(x))
        x = self.pool7(x)

        # 1x1 Conv
        x = F.relu(self.fcn7_1(x))
        x = self.fcn7_2(x)

        # Upsample back to original resolution
        # ERROR FIXED: passing 'x' into interpolate, not 'out'
        out = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

        return out