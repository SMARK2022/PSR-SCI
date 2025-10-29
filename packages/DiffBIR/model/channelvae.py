import torch
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.max_pool = nn.AdaptiveMaxPool2d((2, 2))

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 2, bias=False)

        self.SiLU = nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.SiLU(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.SiLU(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class DoubleConvWoBN(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.double_conv(x) + self.res_conv(x)


class ChannelEncoder(nn.Module):
    def __init__(self):
        super(ChannelEncoder, self).__init__()
        self.conv1 = DoubleConvWoBN(in_channels=28, out_channels=21)
        self.conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            DoubleConvWoBN(in_channels=21, out_channels=9),
        )
        self.conv3 = DoubleConvWoBN(in_channels=9, out_channels=3)
        self.conv_out = DoubleConvWoBN(in_channels=3, out_channels=3)
        self.conv_res = nn.Sequential(
            DoubleConvWoBN(in_channels=28, out_channels=3),
            nn.Upsample(scale_factor=2, mode="bilinear"),
        )

        self.ca1 = ChannelAttention(28, 2)
        self.ca2 = ChannelAttention(21, 2)
        self.ca3 = ChannelAttention(9, 2)
        self.ca_res = ChannelAttention(28, 2)

    def forward(self, x):

        res = self.conv_res(x * self.ca_res(x))

        x = x * self.ca1(x)
        x = self.conv1(x)

        x = x * self.ca2(x)
        x = self.conv2(x)

        x = x * self.ca3(x)
        x = self.conv3(x)

        x = self.conv_out(x + res)
        return x


class ChannelDecoder(nn.Module):
    def __init__(self):
        super(ChannelDecoder, self).__init__()
        self.conv1 = DoubleConvWoBN(in_channels=3, out_channels=9)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=21, kernel_size=2, stride=2),
            nn.SiLU(inplace=True),
            DoubleConvWoBN(in_channels=21, out_channels=21),
        )
        self.conv3 = DoubleConvWoBN(in_channels=21, out_channels=28)
        self.conv_out = DoubleConvWoBN(in_channels=28, out_channels=28)
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=2),
            DoubleConvWoBN(in_channels=3, out_channels=28),
        )

        self.ca3 = ChannelAttention(28, 2)
        self.ca2 = ChannelAttention(21, 2)
        self.ca1 = ChannelAttention(9, 2)

        self.ca_res = ChannelAttention(28, 2)

    def forward(self, x):

        res = self.conv_res(x)
        res = res * self.ca_res(res)

        x = self.conv1(x)
        x = x * self.ca1(x)

        x = self.conv2(x)
        x = x * self.ca2(x)

        x = self.conv3(x)
        x = x * self.ca3(x)

        x = self.conv_out(x + res)

        return x


class ChannelVAE(nn.Module):
    def __init__(self):
        super(ChannelVAE, self).__init__()
        self.encoder = ChannelEncoder()
        self.decoder = ChannelDecoder()

    def forward(self, x):
        en = self.encoder(x)
        return self.decoder(en)
