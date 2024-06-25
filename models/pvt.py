from collections import OrderedDict
import torch
import torch.nn.functional as F
from torchvision.models import resnet
from torch import nn

from models.bb_pvtv2 import pvt_v2_b2, pvt_v2_b1, pvt_v2_b0
from config import Config


config = Config()


class Backbone(nn.Sequential):
    def __init__(self, bb_model):
        super(Backbone, self).__init__()
        self.bb = bb_model
        self.out_channels = config.bb_out_channels[0]

        if config.freeze_bb:
            for key, value in self.named_parameters():
                if 'bb.' in key:
                    value.requires_grad = False

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = self.bb.forward_features_stage1to3(x)[-1]
        return OrderedDict([["feat_res3", feat]])


class Res4Head(nn.Sequential):
    def __init__(self, bb_model):
        super(Res4Head, self).__init__()  # res4
        self.bb = bb_model
        self.out_channels = [config.bb_out_channels[0], config.bb_out_channels[1]]

    def forward(self, x):
        feat = self.bb.forward_features_stage4(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res3", x], ["feat_res4", feat]])


def build_pvt(pvt_weights='b2'):
    if 'b2' in pvt_weights:
        bb_model = pvt_v2_b2()
    elif 'b1' in pvt_weights:
        bb_model = pvt_v2_b1()
    elif 'b0' in pvt_weights:
        bb_model = pvt_v2_b0()
    if config.pvt_weights:
        save_model = torch.load(config.pvt_weights)
        model_dict = bb_model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        bb_model.load_state_dict(model_dict)

    return Backbone(bb_model), Res4Head(bb_model)
