from collections import OrderedDict

import torch
import torch.nn.functional as F
from torchvision import models
from torch import nn
from config import Config
from models.modules import SelfAtt, GAM, SpatialGroupEnhance

from models.bb_pvtv2 import pvt_v2_b2
from config import Config


config = Config()
bb_out_channels = [1024, 512][0]

class Backbone(nn.Sequential):
    def __init__(self):
        super(Backbone, self).__init__()
        self.bb = pvt_v2_b2()
        if config.pvt_weights:
            save_model = torch.load(config.pvt_weights)
            model_dict = self.bb.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.bb.load_state_dict(model_dict)

        self.out_channels = bb_out_channels
        self.conv1x1 = nn.Conv2d(512, bb_out_channels, 1)

        if config.freeze_bb:
            for key, value in self.named_parameters():
                if 'bb.' in key:
                    value.requires_grad = False

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat1, feat2, feat3, feat4 = self.bb(x)
        return OrderedDict([["feat_res4", self.conv1x1(feat4)]])


class Res5Head(nn.Sequential):
    def __init__(self, reid_head="resnet50", pretrained=True):
        super(Res5Head, self).__init__()  # res5
        if reid_head == 'resnet50':
            resnet = models.resnet.__dict__[reid_head](weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif reid_head == 'resnet101':
            resnet = models.resnet.__dict__[reid_head](weights=models.ResNet101_Weights.IMAGENET1K_V2)

        # self.conv1x1 = nn.Conv2d(bb_out_channels, 1024, 1)
        # resnet.layer4[0].conv1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # Change channels_in from 1024 to 512.
        self.res_layer4 = nn.Sequential(OrderedDict([
            # ['conv1x1', self.conv1x1],
            ["layer4", resnet.layer4],
        ]))
        self.out_channels = [bb_out_channels, 2048]

    def forward(self, x):
        # print(1, x.shape)
        # print(list(self.res_layer4.modules())[0])
        # print('*' * 20)
        # exit()
        feat = self.res_layer4(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


def build_pvt(reid_head="resnet50", pretrained=True):
    return Backbone(), Res5Head(reid_head, pretrained)


