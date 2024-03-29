from collections import OrderedDict

import torch.nn.functional as F
from torchvision.models import resnet
from torch import nn
from config import Config


config = Config()

class Backbone(nn.Sequential):
    def __init__(self, bb_model):
        super(Backbone, self).__init__()
        self.bb = nn.Sequential(
            OrderedDict(
                [
                    ["conv1", bb_model.conv1],
                    ["bn1", bb_model.bn1],
                    ["relu", bb_model.relu],
                    ["maxpool", bb_model.maxpool],
                    ["layer1", bb_model.layer1],  # res1
                    ["layer2", bb_model.layer2],  # res2
                    ["layer3", bb_model.layer3],  # res3
                ]
            )
        )

        self.out_channels = config.bb_out_channels[0]

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = self.bb(x)
        return OrderedDict([["feat_res3", feat]])


class Res4Head(nn.Sequential):
    def __init__(self, bb_model):
        super(Res4Head, self).__init__(OrderedDict([["layer4", bb_model.layer4]]))  # res4
        self.out_channels = [config.bb_out_channels[0], config.bb_out_channels[1]]

    def forward(self, x):
        feat = super(Res4Head, self).forward(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res3", x], ["feat_res4", feat]])


def build_resnet50(pretrained=True):
    resnet.model_urls["resnet50"] = "https://download.pytorch.org/models/resnet50-f46c3f97.pth"
    bb_model = resnet.resnet50(pretrained=pretrained if pretrained else None)

    # freeze layers
    bb_model.conv1.weight.requires_grad_(False)
    bb_model.bn1.weight.requires_grad_(False)
    bb_model.bn1.bias.requires_grad_(False)

    return Backbone(bb_model), Res4Head(bb_model)


def build_resnet50_layer4():
    resnet.model_urls["resnet50"] = "https://download.pytorch.org/models/resnet50-f46c3f97.pth"
    resnet50_layer4 = resnet.resnet50(pretrained=True).layer4
    return resnet50_layer4


class MultiPartSpliter(nn.Module):
    def __init__(self, out_channels=None):
        super(MultiPartSpliter, self).__init__()
        self.out_channels = out_channels
        resnet50_layer4_1 = build_resnet50_layer4()
        resnet50_layer4_2 = build_resnet50_layer4()
        resnet50_layer4_3 = build_resnet50_layer4()     # downscale hei and wid as 1/2.
        scales = [1, 2, 3]
        in_feat_size = (14//2, 14//2)     # shape of resnet50_layer4(the output of `box_roi_pool(features, boxes, image_shapes)` in `glcnet.py`).
        self.block_granularity_1 = nn.Sequential(
            resnet50_layer4_1,
            nn.MaxPool2d(kernel_size=(in_feat_size[0] // scales[0], in_feat_size[1])),
        )
        self.block_granularity_2 = nn.Sequential(
            resnet50_layer4_2,
            nn.MaxPool2d(kernel_size=(in_feat_size[0] // scales[1], in_feat_size[1])),
        )
        self.block_granularity_3 = nn.Sequential(
            resnet50_layer4_3,
            nn.MaxPool2d(kernel_size=(in_feat_size[0] // scales[2], in_feat_size[1])),
        )
        if out_channels:
            self.reducer_granularity_1 = nn.Sequential(nn.Conv2d(2048, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
            self.reducer_granularity_2 = nn.Sequential(nn.Conv2d(2048, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
            self.reducer_granularity_3 = nn.Sequential(nn.Conv2d(2048, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
    
    def forward(self, x):
        feat_granularity_1 = self.block_granularity_1(x)
        feat_granularity_2 = self.block_granularity_2(x)
        feat_granularity_3 = self.block_granularity_3(x)
        if self.out_channels:
            feat_granularity_1 = self.reducer_granularity_1(feat_granularity_1)
            feat_granularity_2 = self.reducer_granularity_2(feat_granularity_2)
            feat_granularity_3 = self.reducer_granularity_3(feat_granularity_3)
        feat_granularity_2_horizon_1, feat_granularity_2_horizon_2 = feat_granularity_2[:, :, 0:1, :], feat_granularity_2[:, :, 1:2, :]
        feat_granularity_3_horizon_1, feat_granularity_3_horizon_2, feat_granularity_3_horizon_3 = feat_granularity_3[:, :, 0:1, :], feat_granularity_3[:, :, 1:2, :], feat_granularity_3[:, :, 2:3, :]
        feat_lst = OrderedDict([
            ['feat_granularity_1', feat_granularity_1],
            ['feat_granularity_2_horizon_1', feat_granularity_2_horizon_1], ['feat_granularity_2_horizon_2', feat_granularity_2_horizon_2],
            ['feat_granularity_3_horizon_1', feat_granularity_3_horizon_1], ['feat_granularity_3_horizon_2', feat_granularity_3_horizon_2], ['feat_granularity_3_horizon_3', feat_granularity_3_horizon_3],
        ])
        return feat_lst
