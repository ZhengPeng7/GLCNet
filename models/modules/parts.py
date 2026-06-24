from collections import OrderedDict
from packaging import version
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet, ResNet50_Weights

from configs import config
from models.modules.decoder_blocks import BasicDecBlk
from models.modules.layers import build_norm_layer, _act

torch_version = version.parse(torch.__version__.split("+")[0])
torch_version_legacy = version.parse('1.10.1')


def build_resnet50_layer4(pretrained=True):
    if torch_version <= torch_version_legacy:
        resnet.model_urls["resnet50"] = "https://download.pytorch.org/models/resnet50-f46c3f97.pth"
        resnet50_layer4 = resnet.resnet50(pretrained=pretrained if pretrained else None).layer4
    else:
        resnet50_layer4 = resnet.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None).layer4
    return resnet50_layer4


class MultiPartSpliter(nn.Module):
    def __init__(self, out_channels=None):
        super(MultiPartSpliter, self).__init__()
        self.out_channels = out_channels
        if config.mps_blk == 'BasicDecBlk':
            # BasicDecBlk keeps the resolution.
            inter_channels = 512
            out_channel_mps_blk = config.bb_out_channels[1]
            num_blk = 2
            block_1 = nn.Sequential(
                *[BasicDecBlk(
                    in_channels=config.bb_out_channels[0] if idx_blk == 0 else inter_channels,
                    out_channels=(inter_channels if out_channels else out_channel_mps_blk) if idx_blk == num_blk - 1 else inter_channels
                ) for idx_blk in range(num_blk)]
            )
            block_2 = nn.Sequential(
                *[BasicDecBlk(
                    in_channels=config.bb_out_channels[0] if idx_blk == 0 else inter_channels,
                    out_channels=(inter_channels if out_channels else out_channel_mps_blk) if idx_blk == num_blk - 1 else inter_channels
                ) for idx_blk in range(num_blk)]
            )
            block_3 = nn.Sequential(
                *[BasicDecBlk(
                    in_channels=config.bb_out_channels[0] if idx_blk == 0 else inter_channels,
                    out_channels=(inter_channels if out_channels else out_channel_mps_blk) if idx_blk == num_blk - 1 else inter_channels
                ) for idx_blk in range(num_blk)]
            )
        elif config.mps_blk == 'resnet50_layer4':
            # resnet50_layer4 downscales hei and wid as 1/2
            inter_channels = 2048
            block_1 = nn.Sequential(nn.Conv2d(config.bb_out_channels[0], 1024, 1, 1, 0) if config.bb_out_channels[0] != 1024 else nn.Identity(), build_resnet50_layer4())   # (in_channels, out_channels) of resnet50_layer4 are (1024, 2048).
            block_2 = nn.Sequential(nn.Conv2d(config.bb_out_channels[0], 1024, 1, 1, 0) if config.bb_out_channels[0] != 1024 else nn.Identity(), build_resnet50_layer4())
            block_3 = nn.Sequential(nn.Conv2d(config.bb_out_channels[0], 1024, 1, 1, 0) if config.bb_out_channels[0] != 1024 else nn.Identity(), build_resnet50_layer4())
        num_parts_per_granularity = [1, 2, 3]

        self.block_granularity_1 = nn.Sequential(
            block_1,
            nn.AdaptiveMaxPool2d((num_parts_per_granularity[0], 1)),
        )
        self.block_granularity_2 = nn.Sequential(
            block_2,
            nn.AdaptiveMaxPool2d((num_parts_per_granularity[1], 1)),
        )
        self.block_granularity_3 = nn.Sequential(
            block_3,
            nn.AdaptiveMaxPool2d((num_parts_per_granularity[2], 1)),
        )
        if out_channels:
            self.reducer_granularity_1 = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 1, bias=False), build_norm_layer(out_channels, config.norm_head, eps=1e-3), _act())
            self.reducer_granularity_2 = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 1, bias=False), build_norm_layer(out_channels, config.norm_head, eps=1e-3), _act())
            self.reducer_granularity_3 = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 1, bias=False), build_norm_layer(out_channels, config.norm_head, eps=1e-3), _act())

    def forward(self, x):
        # Apply gradient checkpointing during training if enabled
        if self.training and config.gradient_checkpointing:
            feat_granularity_1 = checkpoint(self.block_granularity_1, x, use_reentrant=False)
            feat_granularity_2 = checkpoint(self.block_granularity_2, x, use_reentrant=False)
            feat_granularity_3 = checkpoint(self.block_granularity_3, x, use_reentrant=False)
        else:
            feat_granularity_1 = self.block_granularity_1(x)
            feat_granularity_2 = self.block_granularity_2(x)
            feat_granularity_3 = self.block_granularity_3(x)
        if self.out_channels:
            feat_granularity_1 = self.reducer_granularity_1(feat_granularity_1)
            feat_granularity_2 = self.reducer_granularity_2(feat_granularity_2)
            feat_granularity_3 = self.reducer_granularity_3(feat_granularity_3)
        # Use split instead of slicing for better memory efficiency
        feat_granularity_2_horizon_1, feat_granularity_2_horizon_2 = feat_granularity_2.split(1, dim=2)
        feat_granularity_3_horizon_1, feat_granularity_3_horizon_2, feat_granularity_3_horizon_3 = feat_granularity_3.split(1, dim=2)
        feat_lst = OrderedDict([
            ['feat_granularity_1', feat_granularity_1],
            ['feat_granularity_2_horizon_1', feat_granularity_2_horizon_1], ['feat_granularity_2_horizon_2', feat_granularity_2_horizon_2],
            ['feat_granularity_3_horizon_1', feat_granularity_3_horizon_1], ['feat_granularity_3_horizon_2', feat_granularity_3_horizon_2], ['feat_granularity_3_horizon_3', feat_granularity_3_horizon_3],
        ])
        return feat_lst
