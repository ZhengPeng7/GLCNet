import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from models.modules import CoAttLayer, ChannelAttention, SpatialAttention
from configs import config


class ContextExtractor1(nn.Module):
    # Attentioned context + x
    def __init__(self, in_channels=config.bb_out_channels[0]):
        super().__init__()
        # Use eps=1e-3 for bf16 compatibility
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            nn.BatchNorm2d(256, eps=1e-3) if config.use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256, eps=1e-3) if config.use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.BatchNorm2d(1, eps=1e-3) if config.use_bn else nn.Identity(),
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        attn = self.convs(x)
        # Compute sigmoid in float32 for numerical stability with bf16/fp16
        attn = torch.sigmoid(attn.float()).to(x.dtype)
        # Compute residual addition in float32 for numerical stability
        x = (x.float() * attn.float() + x.float()).to(x.dtype)
        return self.avgpool(x)


class ContextExtractor2(nn.Module):
    # Attentioned context + x
    def __init__(self, in_channels=config.bb_out_channels[0]):
        super().__init__()
        self.convs = CoAttLayer(in_channels)
        self.avgpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.convs(x)
        return self.avgpool(x)


class ContextExtractor3_scene(nn.Module):
    # Multi-scale featured x + spatial-attentioned context
    def __init__(self, in_channels=config.bb_out_channels[0]):
        super().__init__()
        # Use eps=1e-3 for bf16 compatibility
        self.convs_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, 1, 0), nn.BatchNorm2d(256, eps=1e-3) if config.use_bn else nn.Identity(), nn.ReLU(),
            nn.Conv2d(256, in_channels, 1, 1, 0), nn.BatchNorm2d(in_channels, eps=1e-3) if config.use_bn else nn.Identity(), nn.ReLU(),
        )
        self.convs_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1), nn.BatchNorm2d(256, eps=1e-3) if config.use_bn else nn.Identity(), nn.ReLU(),
            nn.Conv2d(256, in_channels, 3, 1, 1), nn.BatchNorm2d(in_channels, eps=1e-3) if config.use_bn else nn.Identity(), nn.ReLU(),
        )
        self.convs_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 5, 1, 2), nn.BatchNorm2d(256, eps=1e-3) if config.use_bn else nn.Identity(), nn.ReLU(),
            nn.Conv2d(256, in_channels, 5, 1, 2), nn.BatchNorm2d(in_channels, eps=1e-3) if config.use_bn else nn.Identity(), nn.ReLU(),
        )
        self.spatial_attention = SpatialAttention(kernel_size=5)
        self.avgpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        # Apply gradient checkpointing during training if enabled
        if self.training and config.gradient_checkpointing:
            x_1x1 = checkpoint(self.convs_1x1, x, use_reentrant=False)
            x_3x3 = checkpoint(self.convs_3x3, x, use_reentrant=False)
            x_5x5 = checkpoint(self.convs_5x5, x, use_reentrant=False)
        else:
            x_1x1 = self.convs_1x1(x)
            x_3x3 = self.convs_3x3(x)
            x_5x5 = self.convs_5x5(x)
        # Compute feature addition in float32 for numerical stability with bf16/fp16
        x_ms = (x_1x1.float() + x_3x3.float() + x_5x5.float()).to(x.dtype)
        x1 = x_ms * self.spatial_attention(x)
        x = (x1.float() + x.float()).to(x.dtype)
        return self.avgpool(x)


class ContextExtractor3_group(nn.Module):
    # Multi-scale featured x + channel-attentioned context
    def __init__(self, in_channels=config.bb_out_channels[1]):
        super().__init__()
        # Use eps=1e-3 for bf16 compatibility
        self.convs_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, 1, 0), nn.BatchNorm2d(256, eps=1e-3) if config.use_bn else nn.Identity(), nn.ReLU(),
            nn.Conv2d(256, in_channels, 1, 1, 0), nn.BatchNorm2d(in_channels, eps=1e-3) if config.use_bn else nn.Identity(), nn.ReLU(),
        )
        self.convs_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1), nn.BatchNorm2d(256, eps=1e-3) if config.use_bn else nn.Identity(), nn.ReLU(),
            nn.Conv2d(256, in_channels, 3, 1, 1), nn.BatchNorm2d(in_channels, eps=1e-3) if config.use_bn else nn.Identity(), nn.ReLU(),
        )
        self.convs_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 5, 1, 2), nn.BatchNorm2d(256, eps=1e-3) if config.use_bn else nn.Identity(), nn.ReLU(),
            nn.Conv2d(256, in_channels, 5, 1, 2), nn.BatchNorm2d(in_channels, eps=1e-3) if config.use_bn else nn.Identity(), nn.ReLU(),
        )
        self.channel_attention = ChannelAttention(in_channels)
        self.avgpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        # Apply gradient checkpointing during training if enabled
        if self.training and config.gradient_checkpointing:
            x_1x1 = checkpoint(self.convs_1x1, x, use_reentrant=False)
            x_3x3 = checkpoint(self.convs_3x3, x, use_reentrant=False)
            x_5x5 = checkpoint(self.convs_5x5, x, use_reentrant=False)
        else:
            x_1x1 = self.convs_1x1(x)
            x_3x3 = self.convs_3x3(x)
            x_5x5 = self.convs_5x5(x)
        # Compute all operations in float32 for numerical stability with bf16/fp16
        x_ms = x_1x1.float() + x_3x3.float() + x_5x5.float()
        x_out = (x_ms * self.channel_attention(x).float() + x.float()).to(x.dtype)
        return self.avgpool(x_out)

