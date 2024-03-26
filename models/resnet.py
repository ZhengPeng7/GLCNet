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
