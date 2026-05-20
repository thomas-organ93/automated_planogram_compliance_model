import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models


class DirectionNetwork(nn.Module):
    def __init__(self, in_channels=4, internal_channels=128):
        """
        in_channels (int): 4 (RGB + Semantic Mask)
        internal_channels (int): Number of channels for the internal FCN branches.
        """
        super(DirectionNetwork, self).__init__()

        # Load ResNet18 Pretrained backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify First Layer for 4 Channels
        original_conv1 = resnet.conv1
        new_conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        # Disable gradient calculation for weight copying
        with torch.no_grad():
            # Copy RGB weights
            new_conv1.weight[:, :3, :, :] = original_conv1.weight
            new_conv1.weight[:, 3:4, :, :] = torch.mean(original_conv1.weight, dim=1, keepdim=True)

        # Extract specific blocks for skip connections
        self.stem = nn.Sequential(
            new_conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        # Layer 1 (Outputs H/4 = 128 pixels, 64 channels) - Replaces VGG Conv3
        self.layer1 = resnet.layer1
        # Layer 2 (Outputs H/8 = 64 pixels, 128 channels) - Replaces VGG Conv4
        self.layer2 = resnet.layer2
        # Layer 3 (Outputs H/16 = 32 pixels, 256 channels) - Replaces VGG Conv5
        self.layer3 = resnet.layer3

        # Multiscale Aggregation Branches
        self.branch3 = self._make_branch(64, internal_channels)
        self.branch4 = self._make_branch(128, internal_channels)
        self.branch5 = self._make_branch(256, internal_channels)

        # Fusion Block
        self.fusion = nn.Sequential(
            nn.Conv2d(internal_channels * 3, internal_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels, internal_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels, 2, kernel_size=1)  # Output 2 channels (u_x, u_y)
        )

    def _make_branch(self, in_c, out_c):
        """Helper to create the 5x5 -> 1x1 -> 1x1 branch structure."""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=5, padding=2),  # Padding 2 for 5x5 to keep size
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x shape: [Batch, 4, H, W]
        input_size = x.shape[2:]

        # Backbone Forward Pass
        x = self.stem(x)

        # Feature extraction at multiple scales
        c3 = self.layer1(x)  # Shape: H/4, 64 channels
        c4 = self.layer2(c3)  # Shape: H/8, 128 channels
        c5 = self.layer3(c4)  # Shape: H/16, 256 channels

        # Aggregation Branches
        f3 = self.branch3(c3)
        f4 = self.branch4(c4)
        f5 = self.branch5(c5)

        # Upsample lower resolutions to match f3 (H/4)
        f4_up = F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=True)
        f5_up = F.interpolate(f5, size=f3.shape[2:], mode='bilinear', align_corners=True)

        # --- Concatenation ---
        concat = torch.cat([f3, f4_up, f5_up], dim=1)

        # --- Fusion & Final Prediction ---
        out = self.fusion(concat)

        # Upsample to original input resolution
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        return out