import torch.nn as nn

from configs import config
from models.modules.layers import build_norm_layer, _act


class BasicDecBlk(nn.Module):
    """Lightweight decoder block (conv-norm-act -> conv-norm). Simplified from the original
    ASPP-deformable variant; retained as an mps_blk option (config.mps_blk='BasicDecBlk')."""
    def __init__(self, in_channels=64, out_channels=64):
        super(BasicDecBlk, self).__init__()
        inter_channels = 64
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.bn_in = build_norm_layer(inter_channels, config.norm_head, eps=1e-3)
        self.relu_in = _act()
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_out = build_norm_layer(out_channels, config.norm_head, eps=1e-3)

    def forward(self, x):
        x = self.relu_in(self.bn_in(self.conv_in(x)))
        x = self.bn_out(self.conv_out(x))
        return x
