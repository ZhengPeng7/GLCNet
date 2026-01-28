from collections import OrderedDict
from packaging import version
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet, ResNet50_Weights
from torchvision.ops import deform_conv2d
from configs import config
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
            in_feat_size = (14//1, 14//1)     # shape of the output of `box_roi_pool(features, boxes, image_shapes)` in `glcnet.py`.
        elif config.mps_blk == 'resnet50_layer4':
            # resnet50_layer4 downscales hei and wid as 1/2
            inter_channels = 2048
            block_1 = nn.Sequential(nn.Conv2d(config.bb_out_channels[0], 1024, 1, 1, 0) if config.bb_out_channels[0] != 1024 else nn.Identity(), build_resnet50_layer4())   # (in_channels, out_channels) of resnet50_layer4 are (1024, 2048).
            block_2 = nn.Sequential(nn.Conv2d(config.bb_out_channels[0], 1024, 1, 1, 0) if config.bb_out_channels[0] != 1024 else nn.Identity(), build_resnet50_layer4())
            block_3 = nn.Sequential(nn.Conv2d(config.bb_out_channels[0], 1024, 1, 1, 0) if config.bb_out_channels[0] != 1024 else nn.Identity(), build_resnet50_layer4())
            in_feat_size = (14//2, 14//2)
        scales = [1, 2, 3]

        self.block_granularity_1 = nn.Sequential(
            block_1,
            nn.MaxPool2d(kernel_size=(in_feat_size[0] // scales[0], in_feat_size[1])),
        )
        self.block_granularity_2 = nn.Sequential(
            block_2,
            nn.MaxPool2d(kernel_size=(in_feat_size[0] // scales[1], in_feat_size[1])),
        )
        self.block_granularity_3 = nn.Sequential(
            block_3,
            nn.MaxPool2d(kernel_size=(in_feat_size[0] // scales[2], in_feat_size[1])),
        )
        if out_channels:
            # Use eps=1e-3 for bf16 compatibility
            self.reducer_granularity_1 = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels, eps=1e-3) if config.use_bn else nn.Identity(), nn.ReLU())
            self.reducer_granularity_2 = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels, eps=1e-3) if config.use_bn else nn.Identity(), nn.ReLU())
            self.reducer_granularity_3 = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels, eps=1e-3) if config.use_bn else nn.Identity(), nn.ReLU())

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


class BasicDecBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64):
        super(BasicDecBlk, self).__init__()
        inter_channels = [in_channels // 4, 64][1]
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.relu_in = nn.ReLU()
        self.dec_att = ASPPDeformable(in_channels=inter_channels)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        # Use eps=1e-3 for bf16 compatibility
        self.bn_in = nn.BatchNorm2d(inter_channels, eps=1e-3) if config.use_bn else nn.Identity()
        self.bn_out = nn.BatchNorm2d(out_channels, eps=1e-3) if config.use_bn else nn.Identity()

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)
        if hasattr(self, 'dec_att'):
            x = self.dec_att(x)
        x = self.conv_out(x)
        x = self.bn_out(x)
        return x


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        # Compute sigmoid in float32 for better precision with bf16/fp16
        modulator = 2. * torch.sigmoid(self.modulator_conv(x).float()).to(x.dtype)

        x = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
        )
        return x


class _ASPPModuleDeformable(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, padding):
        super(_ASPPModuleDeformable, self).__init__()
        self.atrous_conv = DeformableConv2d(in_channels, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, bias=False)
        # Use eps=1e-3 for bf16 compatibility
        self.bn = nn.BatchNorm2d(planes, eps=1e-3) if config.use_bn else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPPDeformable(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_parallel_block=1):
        super(ASPPDeformable, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale

        self.aspp1 = _ASPPModuleDeformable(in_channels, self.in_channelster, 1, padding=0)
        self.aspp_deforms = nn.ModuleList([
            _ASPPModuleDeformable(in_channels, self.in_channelster, 3, padding=1) for _ in range(num_parallel_block)
        ])

        # Use eps=1e-3 for bf16 compatibility
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels, self.in_channelster, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(self.in_channelster, eps=1e-3) if config.use_bn else nn.Identity(),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(self.in_channelster * (2 + len(self.aspp_deforms)), out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-3) if config.use_bn else nn.Identity()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x_aspp_deforms = [aspp_deform(x) for aspp_deform in self.aspp_deforms]
        x5 = self.global_avg_pool(x)
        # Interpolate in float32 for numerical stability with bf16/fp16
        x5 = nn.functional.interpolate(x5.float(), size=x1.size()[2:], mode='bilinear', align_corners=True).to(x.dtype)
        x = torch.cat((x1, *x_aspp_deforms, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


class CoAttLayer(nn.Module):
    def __init__(self, channel_in=512):
        super(CoAttLayer, self).__init__()

        self.all_attention = GAM(channel_in)
        self.conv_output = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0)
        self.conv_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0)
        self.fc_transform = nn.Linear(channel_in, channel_in)

    def forward(self, x):
        if self.training:
            x_new = self.all_attention(x)
            # Compute mean in float32 for numerical stability with bf16/fp16
            x_proto = torch.mean(x_new.float(), (0, 2, 3), True).to(x.dtype).view(1, -1)
            x_proto = x_proto.unsqueeze(-1).unsqueeze(-1) # 1, C, 1, 1
            weighted_x = x * x_proto
        else:
            x_new = self.all_attention(x)
            # Compute mean in float32 for numerical stability with bf16/fp16
            x_proto = torch.mean(x_new.float(), (0, 2, 3), True).to(x.dtype).view(1, -1)
            x_proto = x_proto.unsqueeze(-1).unsqueeze(-1) # 1, C, 1, 1
            weighted_x = x * x_proto
        return weighted_x


class GAM(nn.Module):
    def __init__(self, channel_in=512):

        super(GAM, self).__init__()
        self.query_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0)
        self.key_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0)

        # Ensure scale is Python float for consistent precision with bf16/fp16
        self.scale = float(1.0 / (channel_in ** 0.5))

        self.conv6 = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x5):
        # x: B,C,H,W
        # x_query: B,C,HW
        B, C, H5, W5 = x5.size()

        x_query = self.query_transform(x5).view(B, C, -1)

        # x_query: B,HW,C
        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C) # BHW, C
        # x_key: B,C,HW
        x_key = self.key_transform(x5).view(B, C, -1)

        x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1) # C, BHW

        # W = Q^T K: B,HW,HW
        # Compute matmul in float32 for numerical stability with bf16/fp16
        x_w = torch.matmul(x_query.float(), x_key.float()) # BHW, BHW
        x_w = x_w.view(B*H5*W5, B, H5*W5)
        x_w = torch.max(x_w, -1).values # BHW, B
        # Explicitly keep mean in float32 to ensure numerical stability
        x_w = x_w.float().mean(-1)
        x_w = x_w.view(B, -1) * self.scale # B, HW
        # Compute softmax in float32 for numerical stability with bf16/fp16
        x_w = nn.functional.softmax(x_w, dim=-1).to(x5.dtype) # B, HW
        x_w = x_w.view(B, H5, W5).unsqueeze(1) # B, 1, H, W

        x5 = x5 * x_w
        x5 = self.conv6(x5)

        return x5


class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        # Compute sigmoid in float32 for numerical stability with bf16/fp16
        y = torch.sigmoid(y.float()).to(x.dtype).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes//ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # Compute addition and sigmoid in float32 for numerical stability with bf16/fp16
        out = (avg_out.float() + max_out.float())
        return torch.sigmoid(out).to(x.dtype)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):
        # Compute mean in float32 for numerical stability with bf16/fp16
        avg_out = x.float().mean(dim=1, keepdim=True).to(x.dtype)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        # Compute sigmoid in float32 for numerical stability
        return torch.sigmoid(out.float()).to(x.dtype)
