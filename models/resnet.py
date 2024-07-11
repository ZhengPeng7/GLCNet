import os
from collections import OrderedDict

import torch
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
    weights_type = ['legacy', 'IMAGENET1K_V1', 'IMAGENET1K_V2'][0]
    bb_resume_custom = [False, 'MovieNet-PS-N{}-ep{}'.format([10, 30, 70][2], [1, 2, 5, 10, 15][0]), 'Pre-trained PS'][0]

    if bb_resume_custom:
        resnet.model_urls["resnet50"] = 'https://download.pytorch.org/models/resnet50-f46c3f97.pth'
        bb_model = resnet.resnet50(pretrained=pretrained if pretrained else None)
        if bb_resume_custom == 'Pre-trained PS':
            # 'https://huggingface.co/Alice10/psvision/resolve/main/resnet50-ps12.pth'
            bb_ckpt_path = os.path.join(os.environ['HOME'], '.cache/torch/hub/checkpoints', 'resnet50-ps12.pth')
        elif 'MovieNet-PS-N' in bb_resume_custom:
            bb_ckpt_path = os.path.join(os.environ['HOME'], 'weights', 'resnet50-pt_mvnps_n{}-ep{}.pth'.format(bb_resume_custom.split('-N')[-1].split('-ep')[0], bb_resume_custom.split('-ep')[-1].split('.pth')[0]))
        bb_model = load_bb_weights(bb_model, bb_ckpt_path)
    else:
        resnet.model_urls["resnet50"] = {
            'legacy': 'https://download.pytorch.org/models/resnet50-f46c3f97.pth',
            'IMAGENET1K_V1': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
            'IMAGENET1K_V2': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        }[weights_type]
        bb_model = resnet.resnet50(pretrained=pretrained if pretrained else None)

    # freeze layers
    bb_model.conv1.weight.requires_grad_(False)
    bb_model.bn1.weight.requires_grad_(False)
    bb_model.bn1.bias.requires_grad_(False)

    return Backbone(bb_model), Res4Head(bb_model)


def build_resnet101(pretrained=True):
    weights_type = ['legacy', 'IMAGENET1K_V1', 'IMAGENET1K_V2'][2]
    resnet.model_urls["resnet101"] = {
        'legacy': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
        'IMAGENET1K_V1': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
        'IMAGENET1K_V2': 'https://download.pytorch.org/models/resnet101-cd907fc2.pth',
    }[weights_type]
    bb_model = resnet.resnet101(pretrained=pretrained if pretrained else None)

    # freeze layers
    bb_model.conv1.weight.requires_grad_(False)
    bb_model.bn1.weight.requires_grad_(False)
    bb_model.bn1.bias.requires_grad_(False)

    return Backbone(bb_model), Res4Head(bb_model)


def load_bb_weights(bb_model, bb_ckpt_path):
    save_model = torch.load(bb_ckpt_path, map_location='cpu')
    if 'model' in save_model.keys():
        save_model = save_model['model']
        save_model = {k.lstrip('backbone.bb.'): v for k, v in save_model.items()}
    model_dict = bb_model.state_dict()
    state_dict = {k: v if v.size() == model_dict[k].size() else model_dict[k] for k, v in save_model.items() if k in model_dict.keys()}
    # to ignore the weights with mismatched size when I modify the backbone itself.
    if not state_dict:
        save_model_keys = list(save_model.keys())
        sub_item = save_model_keys[0] if len(save_model_keys) == 1 else None
        state_dict = {k: v if v.size() == model_dict[k].size() else model_dict[k] for k, v in save_model[sub_item].items() if k in model_dict.keys()}
        if not state_dict or not sub_item:
            print('Weights are not successully loaded. Check the state dict of weights file.')
            return None
        else:
            print('Found correct weights in the "{}" item of loaded state_dict.'.format(sub_item))
    model_dict.update(state_dict)
    bb_model.load_state_dict(model_dict)
    return bb_model
