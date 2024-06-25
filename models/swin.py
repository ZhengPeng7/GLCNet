from collections import OrderedDict
import torch
import torch.nn.functional as F
from torchvision.models import resnet
from torch import nn

from models.bb_swin import swin_v1_t, swin_v1_s, swin_v1_b
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
        feat = self.bb.forward_features_stage1to3(x)
        return OrderedDict([["feat_res3", feat[-2]], ['x_for_stage_4', feat[-1]]])


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


def build_swin(weights='_t'):
    if '_tiny' in weights:
        bb_model = swin_v1_t()
    elif '_small' in weights:
        bb_model = swin_v1_s()
    elif '_base' in weights:
        bb_model = swin_v1_b()
    if weights:
        save_model = torch.load(weights, map_location='cpu')
        model_dict = bb_model.state_dict()
        save_model_keys = list(save_model.keys())
        sub_item = save_model_keys[0] if len(save_model_keys) == 1 else None
        state_dict = {k: v if v.size() == model_dict[k].size() else model_dict[k] for k, v in save_model[sub_item].items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        bb_model.load_state_dict(model_dict)

    return Backbone(bb_model), Res4Head(bb_model)
