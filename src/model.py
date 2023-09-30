import torch
import torch.nn as nn
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        bias=True,
        batch_norm=False,
    ):
        super().__init__()
        if batch_norm:
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
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=bias,
                ),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class DoubleConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, batch_norm=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ConvBlock(in_channels, mid_channels, batch_norm=batch_norm),
            ConvBlock(mid_channels, out_channels, batch_norm=batch_norm),
        )

    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), ConvBlock(in_channels, out_channels, batch_norm=batch_norm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first=False, batch_norm=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=2, stride=2
        )
        if first:
            self.conv = ConvBlock(in_channels, out_channels, batch_norm=batch_norm)
        else:
            self.conv = DoubleConvBlock(
                in_channels, out_channels, batch_norm=batch_norm
            )

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.up(x1)
        return x2


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super(UNet, self).__init__()
        self.init_conv = DoubleConvBlock(in_channels, 32, batch_norm=batch_norm)
        self.encoder1 = EncoderBlock(32, 32, batch_norm=batch_norm)
        self.encoder2 = EncoderBlock(32, 32, batch_norm=batch_norm)
        self.encoder3 = EncoderBlock(32, 32, batch_norm=batch_norm)
        self.encoder4 = EncoderBlock(32, 32, batch_norm=batch_norm)
        self.encoder5 = EncoderBlock(32, 32, batch_norm=batch_norm)
        self.decoder1 = DecoderBlock(32, 32, first=True, batch_norm=batch_norm)
        self.decoder2 = DecoderBlock(64, 64, batch_norm=batch_norm)
        self.decoder3 = DecoderBlock(96, 64, batch_norm=batch_norm)
        self.decoder4 = DecoderBlock(96, 64, batch_norm=batch_norm)
        self.decoder5 = DecoderBlock(96, 64, batch_norm=batch_norm)
        self.last = nn.Sequential(
            DoubleConvBlock(96, 64),  # (96, 384, 384) -> (64, 384, 384)
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x0):
        x1 = self.init_conv(x0)  # (in_channels, 384, 384) -> (32, 384, 384)
        x2 = self.encoder1(x1)  # (32, 384, 384) -> (32, 192, 192)
        x3 = self.encoder2(x2)  # (32, 192, 192) -> (32, 96, 96)
        x4 = self.encoder3(x3)  # (32, 96, 96) -> (32, 48, 48)
        x5 = self.encoder4(x4)  # (32, 48, 48) -> (32, 24, 24)
        x6 = self.encoder5(x5)  # (32, 24, 24) -> (32, 12, 12)
        x = self.decoder1(x6)  # (32, 12, 12) -> (32, 24, 24)
        x = self.decoder2(torch.cat([x5, x], dim=1))  # (64, 24, 24) -> (64, 48, 48)
        x = self.decoder3(torch.cat([x4, x], dim=1))  # (96, 48, 48) -> (64, 96, 96)
        x = self.decoder4(torch.cat([x3, x], dim=1))  # (96, 96, 96) -> (64, 192, 192)
        x = self.decoder5(torch.cat([x2, x], dim=1))  # (96, 192, 192) -> (64, 384, 384)
        return self.last(
            torch.cat([x1, x], dim=1)
        )  # (96, 384, 384) -> (out_channels, 384, 384)


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
    summary(model, (3, 384, 384), device="cpu")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
