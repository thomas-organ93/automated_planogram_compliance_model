import torch
import torch.nn as nn


def double_convolution(in_channels, out_channels):
    """
    (Conv -> Group Normalisation -> Leaky ReLU) * 2
    # Removed Batch Normalisation for Group Normalisation because of small batch size (4)
    # Group Norm = 32, creates finer sub-feature groups. https://arxiv.org/abs/1803.08494
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=32, num_channels=out_channels),  # Added Group Norm
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=32, num_channels=out_channels),  # Added Group Norm
        nn.LeakyReLU(0.2, inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(UNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path (Encoder)
        self.down_conv_1 = double_convolution(in_channels, 64)
        self.down_conv_2 = double_convolution(64, 128)
        self.down_conv_3 = double_convolution(128, 256)
        self.down_conv_4 = double_convolution(256, 512)
        self.down_conv_5 = double_convolution(512, 1024)

        # Expanding path (Decoder)
        self.up_transpose_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv_1 = double_convolution(1024, 512)

        self.up_transpose_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_2 = double_convolution(512, 256)

        self.up_transpose_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_3 = double_convolution(256, 128)

        self.up_transpose_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv_4 = double_convolution(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        # We save the output of the convs as 'skip' vars to use later
        skip1 = self.down_conv_1(x)
        x = self.max_pool2d(skip1)

        skip2 = self.down_conv_2(x)
        x = self.max_pool2d(skip2)

        skip3 = self.down_conv_3(x)
        x = self.max_pool2d(skip3)

        skip4 = self.down_conv_4(x)
        x = self.max_pool2d(skip4)

        # Bottleneck (no pooling after this)
        x = self.down_conv_5(x)

        # Decoder
        x = self.up_transpose_1(x)
        x = torch.cat([skip4, x], dim=1)
        x = self.up_conv_1(x)

        x = self.up_transpose_2(x)
        x = torch.cat([skip3, x], dim=1)
        x = self.up_conv_2(x)

        x = self.up_transpose_3(x)
        x = torch.cat([skip2, x], dim=1)
        x = self.up_conv_3(x)

        x = self.up_transpose_4(x)
        x = torch.cat([skip1, x], dim=1)
        x = self.up_conv_4(x)

        return self.out(x)


if __name__ == '__main__':
    # Test with random data
    input_image = torch.rand((1, 3, 512, 512))
    model = UNet(in_channels=3, num_classes=10)

    outputs = model(input_image)
    print(f"Input Shape: {input_image.shape}")
    print(f"Output Shape: {outputs.shape}")