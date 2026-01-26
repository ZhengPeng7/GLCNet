from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import config


class SwinBackbone(nn.Module):
    """Wrapper for Swin Transformer backbone to match GLCNet interface."""
    def __init__(self, swin_model):
        super().__init__()
        self.bb = swin_model
        self.out_channels = config.bb_out_channels[0]

        if config.freeze_bb:
            for param in self.bb.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Swin outputs tuple of features from each stage
        # Stage indices: 0, 1, 2, 3 correspond to different resolutions
        # For GLCNet, we use stage 1 output as feat_res3 (similar resolution to ResNet layer3)
        feats = self.bb(x)
        feat_res3 = feats[1]  # Stage 1 output (stride 8)
        return OrderedDict([["feat_res3", feat_res3]])


class SwinRes4Head(nn.Module):
    """Wrapper for Swin Transformer head to match GLCNet interface."""
    def __init__(self, swin_model):
        super().__init__()
        self.bb = swin_model
        self.out_channels = [config.bb_out_channels[0], config.bb_out_channels[1]]
        # Store stage 2 and 3 features for later use
        self._feat_cache = None

    def forward(self, x):
        # x is already feat_res3 from SwinBackbone
        # We need to get feat_res4 from the cached features
        # Since Swin processes all stages together, we cache during backbone forward
        # Here x is the input that should be processed further
        x_pooled = F.adaptive_max_pool2d(x, 1)

        # For Swin, the res4 features are stored in _feat_cache during backbone forward
        if self._feat_cache is not None:
            feat_res4 = self._feat_cache
            feat_res4_pooled = F.adaptive_max_pool2d(feat_res4, 1)
        else:
            # Fallback: use input as res4 (shouldn't happen normally)
            feat_res4_pooled = x_pooled

        return OrderedDict([["feat_res3", x_pooled], ["feat_res4", feat_res4_pooled]])


class SwinBackboneFull(nn.Module):
    """Full Swin backbone that provides both feat_res3 and caches feat_res4."""
    def __init__(self, swin_model, out_channels):
        super().__init__()
        self.bb = swin_model
        self.out_channels = out_channels[0]
        self._out_channels_all = out_channels
        self.feat_res4_cache = None

        if config.freeze_bb:
            for param in self.bb.parameters():
                param.requires_grad = False

    def forward(self, x):
        feats = self.bb(x)
        # Stage 2 (stride 16) as feat_res3, Stage 3 (stride 32) as feat_res4
        # This matches ResNet layer3/layer4 spatial resolution
        feat_res3 = feats[2]  # Stage 2 output
        self.feat_res4_cache = feats[3]  # Stage 3 output (cache for Res4Head)
        return OrderedDict([["feat_res3", feat_res3]])


class SwinRes4HeadWithCache(nn.Module):
    """Swin head that processes RoI features to produce feat_res4."""
    def __init__(self, backbone_ref):
        super().__init__()
        self.backbone_ref = backbone_ref
        self.out_channels = backbone_ref._out_channels_all
        in_channels = self.out_channels[0]
        out_channels = self.out_channels[1]
        # Conv block to transform RoI features from res3 channels to res4 channels
        # Similar to ResNet's layer4 processing
        self.res4_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (num_proposals, res3_channels, 14, 14) - RoI-pooled features
        x_pooled = F.adaptive_max_pool2d(x, 1)
        # Process RoI features through conv block to get res4-level features
        feat_res4 = self.res4_conv(x)
        feat_res4_pooled = F.adaptive_max_pool2d(feat_res4, 1)
        return OrderedDict([["feat_res3", x_pooled], ["feat_res4", feat_res4_pooled]])


class DinoBackbone(nn.Module):
    """Wrapper for DINOv3 backbone to match GLCNet interface."""
    def __init__(self, dino_model, out_channels):
        super().__init__()
        self.bb = dino_model
        self.out_channels = out_channels[0]
        self._out_channels_all = out_channels
        self.feat_res4_cache = None

        if config.freeze_bb:
            for param in self.bb.parameters():
                param.requires_grad = False

    def forward(self, x):
        # DINOv3 outputs features from different transformer layers
        # All have same spatial resolution (based on patch size)
        feats = self.bb(x)
        # Use second-to-last feature as feat_res3, last as feat_res4
        feat_res3 = feats[2]  # Third output
        self.feat_res4_cache = feats[3]  # Fourth output (cache for head)
        return OrderedDict([["feat_res3", feat_res3]])


class DinoRes4Head(nn.Module):
    """DINOv3 head that processes RoI features to produce feat_res4."""
    def __init__(self, backbone_ref):
        super().__init__()
        self.backbone_ref = backbone_ref
        self.out_channels = backbone_ref._out_channels_all
        in_channels = self.out_channels[0]
        out_channels = self.out_channels[1]
        # Conv block to transform RoI features
        # For DINOv3, res3 and res4 have same channels, so this is identity-like
        if in_channels == out_channels:
            self.res4_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-3),
                nn.ReLU(inplace=True),
            )
        else:
            self.res4_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-3),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-3),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        # x: (num_proposals, res3_channels, 14, 14) - RoI-pooled features
        x_pooled = F.adaptive_max_pool2d(x, 1)
        # Process RoI features through conv block to get res4-level features
        feat_res4 = self.res4_conv(x)
        feat_res4_pooled = F.adaptive_max_pool2d(feat_res4, 1)
        return OrderedDict([["feat_res3", x_pooled], ["feat_res4", feat_res4_pooled]])


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
    state_dict = {k: v if v.size() == model_dict[k].size() else model_dict[k]
                  for k, v in save_model.items() if k in model_dict.keys()}

    if not state_dict:
        save_model_keys = list(save_model.keys())
        sub_item = save_model_keys[0] if len(save_model_keys) == 1 else None
        if sub_item and isinstance(save_model[sub_item], dict):
            state_dict = {k: v if v.size() == model_dict[k].size() else model_dict[k]
                          for k, v in save_model[sub_item].items() if k in model_dict.keys()}
        if not state_dict:
            print(f'Weights are not successfully loaded for {model_name}. Check the state dict.')
            return model
        else:
            print(f'Found correct weights in the "{sub_item}" item of loaded state_dict.')

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    print(f'Loaded weights for {model_name} from {weights_path}')
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
        box_head = SwinRes4HeadWithCache(backbone)
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

        dino_model = dino_builders[bb_name]()
        if pretrained:
            dino_model = load_weights(dino_model, bb_name)

        out_channels = config.bb_out_channels_collection.get(bb_name, [1024, 1024])
        backbone = DinoBackbone(dino_model, out_channels)
        box_head = DinoRes4Head(backbone)
        return backbone, box_head

    else:
        raise ValueError(f'Unknown backbone: {bb_name}')
