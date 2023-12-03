import torch
import torch.nn as nn


class ContextExtractorSceneConvs1(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1024):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1)
        )

    def forward(self, x):
        return self.convs(x)


class ContextExtractorSceneConvs2(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1024):
        super().__init__()
        self.attn_convs = nn.Sequential(
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
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        attn = self.convs(x)
        x = attn * self.convs(x) + x
        return self.avgpool(x)



class ContextExtractorGroupConvs1(nn.Module):
    def __init__(self, in_channels=2048, out_channels=2048):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1)
        )

    def forward(self, x):
        return self.convs(x)


class ContextExtractorGroupConvs2(nn.Module):
    def __init__(self, in_channels=2048, out_channels=2048):
        super().__init__()
        self.attn_convs = nn.Sequential(
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
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        attn = self.convs(x)
        x = attn * self.convs(x) + x
        return self.avgpool(x)
