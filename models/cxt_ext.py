import torch
import torch.nn as nn
from models.modules import CoAttLayer


class ContextExtractor1(nn.Module):
    # Attentioned context + x
    def __init__(self, in_channels=1024, out_channels=None):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
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
    def __init__(self, in_channels=1024, out_channels=None):
        super().__init__()
        self.convs = CoAttLayer(in_channels)
        self.avgpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.convs(x)
        return self.avgpool(x)
