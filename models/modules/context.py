import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from models.modules import ChannelAttention
from models.modules.layers import build_norm_layer
from configs import config


class ContextExtractor1(nn.Module):
    # Attentioned context + x
    def __init__(self, in_channels=config.bb_out_channels[0]):
        super().__init__()
        # config.norm_context: 'BN' baseline / 'GN' batch-independent / 'none'. eps=1e-3 for bf16.
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            build_norm_layer(256, config.norm_context, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            build_norm_layer(256, config.norm_context, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(256, 1, 3, 1, 1),
            build_norm_layer(1, config.norm_context, eps=1e-3),
        )
        self.avgpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        attn = self.convs(x)
        # Compute sigmoid in float32 for numerical stability with bf16/fp16
        attn = torch.sigmoid(attn.float()).to(x.dtype)
        # Compute residual addition in float32 for numerical stability
        x = (x.float() * attn.float() + x.float()).to(x.dtype)
        return self.avgpool(x)


class ContextExtractor3_group(nn.Module):
    # Multi-scale featured x + channel-attentioned context
    def __init__(self, in_channels=config.bb_out_channels[1]):
        super().__init__()
        # config.norm_context: 'BN' baseline / 'GN' batch-independent / 'none'. eps=1e-3 for bf16.
        self.convs_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, 1, 0), build_norm_layer(256, config.norm_context, eps=1e-3), nn.ReLU(),
            nn.Conv2d(256, in_channels, 1, 1, 0), build_norm_layer(in_channels, config.norm_context, eps=1e-3), nn.ReLU(),
        )
        self.convs_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1), build_norm_layer(256, config.norm_context, eps=1e-3), nn.ReLU(),
            nn.Conv2d(256, in_channels, 3, 1, 1), build_norm_layer(in_channels, config.norm_context, eps=1e-3), nn.ReLU(),
        )
        self.convs_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 5, 1, 2), build_norm_layer(256, config.norm_context, eps=1e-3), nn.ReLU(),
            nn.Conv2d(256, in_channels, 5, 1, 2), build_norm_layer(in_channels, config.norm_context, eps=1e-3), nn.ReLU(),
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
