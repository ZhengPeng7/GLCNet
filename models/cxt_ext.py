import torch
import torch.nn as nn
from models.modules import CoAttLayer, ChannelAttention, SpatialAttention
from config import Config


config = Config()


class ContextExtractor1(nn.Module):
    # Attentioned context + x
    def __init__(self, in_channels=1024):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            nn.BatchNorm2d(256) if config.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256) if config.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.BatchNorm2d(1) if config.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        attn = self.convs(x)
        x = x * attn + x
        return self.avgpool(x)


class ContextExtractor2(nn.Module):
    # Attentioned context + x
    def __init__(self, in_channels=1024):
        super().__init__()
        self.convs = CoAttLayer(in_channels)
        self.avgpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.convs(x)
        return self.avgpool(x)


class ContextExtractor3_scene(nn.Module):
    # Multi-scale featured x + spatial-attentioned context
    def __init__(self, in_channels=1024):
        super().__init__()
        self.convs_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, 1, 0), nn.BatchNorm2d(256) if config.use_bn else nn.Identity(), nn.ReLU(inplace=True),
            nn.Conv2d(256, in_channels, 1, 1, 0), nn.BatchNorm2d(in_channels) if config.use_bn else nn.Identity(), nn.ReLU(inplace=True),
        )
        self.convs_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1), nn.BatchNorm2d(256) if config.use_bn else nn.Identity(), nn.ReLU(inplace=True),
            nn.Conv2d(256, in_channels, 3, 1, 1), nn.BatchNorm2d(in_channels) if config.use_bn else nn.Identity(), nn.ReLU(inplace=True),
        )
        self.convs_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 5, 1, 2), nn.BatchNorm2d(256) if config.use_bn else nn.Identity(), nn.ReLU(inplace=True),
            nn.Conv2d(256, in_channels, 5, 1, 2), nn.BatchNorm2d(in_channels) if config.use_bn else nn.Identity(), nn.ReLU(inplace=True),
        )
        self.spatial_attention = SpatialAttention(kernel_size=5)
        self.avgpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x_1x1 = self.convs_1x1(x)
        x_3x3 = self.convs_3x3(x)
        x_5x5 = self.convs_5x5(x)
        # x_ms = torch.cat([x_1x1, x_3x3, x_5x5], dim=1)
        x_ms = x_1x1 + x_3x3 + x_5x5
        x1 = x_ms * self.spatial_attention(x)
        x = x1 + x
        return self.avgpool(x)


class ContextExtractor3_group(nn.Module):
    # Multi-scale featured x + channel-attentioned context
    def __init__(self, in_channels=1024):
        super().__init__()
        self.convs_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, 1, 0), nn.BatchNorm2d(256) if config.use_bn else nn.Identity(), nn.ReLU(inplace=True),
            nn.Conv2d(256, in_channels, 1, 1, 0), nn.BatchNorm2d(in_channels) if config.use_bn else nn.Identity(), nn.ReLU(inplace=True),
        )
        self.convs_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1), nn.BatchNorm2d(256) if config.use_bn else nn.Identity(), nn.ReLU(inplace=True),
            nn.Conv2d(256, in_channels, 3, 1, 1), nn.BatchNorm2d(in_channels) if config.use_bn else nn.Identity(), nn.ReLU(inplace=True),
        )
        self.convs_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 5, 1, 2), nn.BatchNorm2d(256) if config.use_bn else nn.Identity(), nn.ReLU(inplace=True),
            nn.Conv2d(256, in_channels, 5, 1, 2), nn.BatchNorm2d(in_channels) if config.use_bn else nn.Identity(), nn.ReLU(inplace=True),
        )
        self.channel_attention = ChannelAttention(in_channels)
        self.avgpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x_1x1 = self.convs_1x1(x)
        x_3x3 = self.convs_3x3(x)
        x_5x5 = self.convs_5x5(x)
        # x_ms = torch.cat([x_1x1, x_3x3, x_5x5], dim=1)
        x_ms = x_1x1 + x_3x3 + x_5x5
        x = x_ms * self.channel_attention(x) + x
        return self.avgpool(x)

