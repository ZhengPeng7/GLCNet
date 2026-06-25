from collections import OrderedDict
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import config
from models.modules.layers import build_norm_layer


class SwinBackboneFull(nn.Module):
    """Full Swin backbone that provides feat_res3 (the deepest stage is fabricated by the Res4Head)."""
    def __init__(self, swin_model, out_channels):
        super().__init__()
        self.bb = swin_model
        self.out_channels = out_channels[0]
        self._out_channels_all = out_channels

        if config.freeze_bb:
            for param in self.bb.parameters():
                param.requires_grad = False

    def forward(self, x):
        feats = self.bb(x)
        assert len(feats) >= 3, f'backbone must expose >=3 feature stages, got {len(feats)}'
        # Stage 2 (stride 16) as feat_res3 (matches ResNet layer3 resolution); feat_res4 is fabricated
        # from RoI-pooled feat_res3 by the Res4Head (see SwinRes4Head).
        return OrderedDict([["feat_res3", feats[2]]])


class SwinRes4Head(nn.Module):
    """Swin head that fabricates feat_res4 (res4-level) from RoI-pooled feat_res3, like ResNet's layer4."""
    def __init__(self, backbone_ref):
        super().__init__()
        self.out_channels = backbone_ref._out_channels_all
        in_channels = self.out_channels[0]
        out_channels = self.out_channels[1]
        self.res4_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            build_norm_layer(out_channels, config.norm_res4, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            build_norm_layer(out_channels, config.norm_res4, eps=1e-3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (num_proposals, res3_channels, 14, 14) - RoI-pooled features
        x_pooled = F.adaptive_max_pool2d(x, 1)
        feat_res4_pooled = F.adaptive_max_pool2d(self.res4_conv(x), 1)
        return OrderedDict([["feat_res3", x_pooled], ["feat_res4", feat_res4_pooled]])


class DinoBackbone(nn.Module):
    """Wrapper for DINOv3 backbone to match GLCNet's one-feature RPN interface."""
    def __init__(self, dino_model, out_channels):
        super().__init__()
        self.bb = dino_model
        self.out_channels = out_channels[0]
        self._out_channels_all = out_channels
        self.feat_res4_stage = None
        self.use_stage4_roi = config.dino_v3_stage_aligned_features_active
        self.feat_res3_output_pos = config.dino_v3_feat_res3_output_pos
        self.feat_res4_output_pos = config.dino_v3_feat_res4_output_pos

        if config.freeze_bb:
            self._set_backbone_trainability()

    def _set_backbone_trainability(self):
        trainable_blocks = set(config.dino_v3_trainable_block_ids)
        for name, param in self.bb.named_parameters():
            block_match = re.search(r'(?:^|\.)blocks\.(\d+)\.', name)
            param.requires_grad = bool(
                block_match and int(block_match.group(1)) in trainable_blocks
            )

    def forward(self, x):
        self.feat_res4_stage = None
        feats = self.bb(x)
        assert len(feats) >= 1, f'backbone must expose >=1 feature stage, got {len(feats)}'
        feat_res3 = feats[self.feat_res3_output_pos]
        if self.use_stage4_roi:
            assert self.feat_res4_output_pos is not None
            assert len(feats) > self.feat_res4_output_pos, (
                f'DINOv3 stage4 feature missing: got {len(feats)} outputs'
            )
            self.feat_res4_stage = feats[self.feat_res4_output_pos]
        return OrderedDict([["feat_res3", feat_res3]])


class DinoRes4Head(nn.Module):
    """DINOv3 head that consumes true stage4 RoIs or fabricates them in legacy mode."""
    def __init__(self, backbone_ref):
        super().__init__()
        self.out_channels = backbone_ref._out_channels_all
        in_channels = self.out_channels[0]
        out_channels = self.out_channels[1]
        self.use_stage4_roi = config.dino_v3_stage_aligned_features_active
        if self.use_stage4_roi:
            self.res4_conv = None
        else:
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                build_norm_layer(out_channels, config.norm_res4, eps=1e-3),
                nn.ReLU(inplace=True),
            ]
            if in_channels != out_channels:
                layers += [
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    build_norm_layer(out_channels, config.norm_res4, eps=1e-3),
                    nn.ReLU(inplace=True),
                ]
            self.res4_conv = nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            feat_res3_roi, feat_res4_roi = x
            x_pooled = F.adaptive_max_pool2d(feat_res3_roi, 1)
            feat_res4_pooled = F.adaptive_max_pool2d(feat_res4_roi, 1)
            return OrderedDict([["feat_res3", x_pooled], ["feat_res4", feat_res4_pooled]])
        if self.use_stage4_roi:
            raise RuntimeError('DINOv3 stage-aligned head expects paired stage3/stage4 RoI features.')
        x_pooled = F.adaptive_max_pool2d(x, 1)
        feat_res4_pooled = F.adaptive_max_pool2d(self.res4_conv(x), 1)
        return OrderedDict([["feat_res3", x_pooled], ["feat_res4", feat_res4_pooled]])


_BB_LOAD_MIN_MATCH_FRAC = 0.5  # refuse to "load" a mostly-random backbone


def _filter_matching(source, model_dict):
    """Split source state_dict vs model_dict into matched (same shape) / mismatched / unexpected keys."""
    matched, mismatched, unexpected = {}, [], []
    for k, v in source.items():
        if k not in model_dict:
            unexpected.append(k)
        elif v.size() == model_dict[k].size():
            matched[k] = v
        else:
            mismatched.append(k)
    return matched, mismatched, unexpected


def load_weights(model, model_name):
    """Load pretrained weights for a model."""
    weights_path = config.weights.get(model_name)
    if not weights_path:
        print(f'No weights path configured for {model_name}')
        return model

    try:
        save_model = torch.load(weights_path, map_location='cpu', weights_only=True)
    except FileNotFoundError:
        print(f'Weights file not found: {weights_path}')
        return model

    model_dict = model.state_dict()
    matched, mismatched, unexpected = _filter_matching(save_model, model_dict)
    if not matched:
        # Some checkpoints wrap the real state_dict under a single top-level key.
        keys = list(save_model.keys())
        sub_item = keys[0] if len(keys) == 1 else None
        if sub_item and isinstance(save_model[sub_item], dict):
            matched, mismatched, unexpected = _filter_matching(save_model[sub_item], model_dict)
            if matched:
                print(f'Found weights in the "{sub_item}" item of loaded state_dict.')

    match_frac = len(matched) / max(len(model_dict), 1)
    if match_frac < _BB_LOAD_MIN_MATCH_FRAC:
        raise RuntimeError(
            f'Refusing to load {model_name} from {weights_path}: only {len(matched)}/{len(model_dict)} '
            f'({match_frac:.1%}) weights matched ({len(mismatched)} shape-mismatched, '
            f'{len(unexpected)} unexpected) — the backbone would be mostly randomly initialized.'
        )

    model_dict.update(matched)
    model.load_state_dict(model_dict)
    print(f'Loaded {len(matched)}/{len(model_dict)} weights for {model_name} from {weights_path} '
          f'({len(mismatched)} shape-mismatched, {len(unexpected)} unexpected keys).')
    return model


def build_backbone(bb_name, pretrained=True):
    """
    Build backbone and head for GLCNet.

    Returns:
        tuple: (backbone, box_head) where:
            - backbone: has out_channels (int), forward returns OrderedDict with feat_res3
            - box_head: has out_channels ([int, int]), forward returns OrderedDict with feat_res3 and feat_res4
    """
    if bb_name == 'resnet50':
        from models.backbones.resnet import build_resnet50
        return build_resnet50(pretrained=pretrained)

    elif bb_name == 'resnet101':
        from models.backbones.resnet import build_resnet101
        return build_resnet101(pretrained=pretrained)

    elif bb_name.startswith('pvt_v2'):
        from models.backbones.pvt_v2 import build_pvt
        return build_pvt(weights=config.weights.get(bb_name, ''), bb_name=bb_name)

    elif bb_name.startswith('swin_v1'):
        from models.backbones.swin_v1 import swin_v1_t, swin_v1_s, swin_v1_b, swin_v1_l
        swin_builders = {
            'swin_v1_t': swin_v1_t,
            'swin_v1_s': swin_v1_s,
            'swin_v1_b': swin_v1_b,
            'swin_v1_l': swin_v1_l,
        }
        if bb_name not in swin_builders:
            raise ValueError(f'Unknown Swin variant: {bb_name}')

        swin_model = swin_builders[bb_name]()
        if pretrained:
            swin_model = load_weights(swin_model, bb_name)

        out_channels = config.bb_out_channels_collection.get(bb_name, [1024, 2048])
        backbone = SwinBackboneFull(swin_model, out_channels)
        box_head = SwinRes4Head(backbone)
        return backbone, box_head

    elif bb_name.startswith('dino_v3'):
        from models.backbones.dino_v3 import dino_v3_b, dino_v3_l, dino_v3_h_plus, dino_v3_7b
        dino_builders = {
            'dino_v3_b': dino_v3_b,
            'dino_v3_l': dino_v3_l,
            'dino_v3_h_plus': dino_v3_h_plus,
            'dino_v3_7b': dino_v3_7b,
        }
        if bb_name not in dino_builders:
            raise ValueError(f'Unknown DINOv3 variant: {bb_name}')

        dino_model = dino_builders[bb_name](out_indices=config.dino_v3_out_indices)
        if pretrained:
            dino_model = load_weights(dino_model, bb_name)

        out_channels = config.bb_out_channels_collection.get(bb_name, [1024, 1024])
        backbone = DinoBackbone(dino_model, out_channels)
        box_head = DinoRes4Head(backbone)
        return backbone, box_head

    else:
        raise ValueError(f'Unknown backbone: {bb_name}')
