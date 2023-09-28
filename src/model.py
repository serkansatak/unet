import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ConvBlock(in_channels, mid_channels), ConvBlock(mid_channels, out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=2, stride=2
        )
        if first:
            self.conv = ConvBlock(in_channels, out_channels)
        else:
            self.conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.up(x1)
        return x2


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.init_conv = DoubleConvBlock(in_channels, 32)
        self.encoder = EncoderBlock(32, 32)
        self.decoder1 = DecoderBlock(32, 32, first=True)
        self.decoder2 = DecoderBlock(64, 64)
        self.decoder3 = DecoderBlock(96, 64)
        self.last = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.init_conv(x)
        x2 = self.encoder(x1)
        x3 = self.encoder(x2)
        x4 = self.encoder(x3)
        x5 = self.encoder(x4)
        x6 = self.encoder(x5)
        x = self.decoder1(x6)
        x = self.decoder2(torch.cat([x, x5], dim=1))
        x = self.decoder3(torch.cat([x, x4], dim=1))
        x = self.decoder3(torch.cat([x, x3], dim=1))
        x = self.decoder3(torch.cat([x, x2], dim=1))
        return self.last(x)


model_map = {
    "unet": UNet,
}


def get_model(model_name: str, in_channels: int, out_channels: int) -> nn.Module:
    if model_name not in model_map.keys():
        raise ValueError(f"Unknown model name {model_name}")
    return model_map[model_name](in_channels, out_channels)


if __name__ == "__main__":
    model = UNet(3, 3)
    x = torch.randn(1, 3, 384, 384)
    print(model(x).shape)
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
