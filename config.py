import math
from defaults import get_default_cfg


cfg = get_default_cfg()


class Config():
    def __init__(self) -> None:
        # Context Features
        self.cxt_scene_enabled = True
        self.cxt_group_enabled = False
        self.cxt_group_labelledOnly = False

        self.cxt_ext_scene = [0, 1, 2, 3, 4][1]     # [1] is the best one.
        self.cxt_ext_group = [0, 1, 2, 3, 4][1]     # [1] is the best one.
        self.lr = 0.003 * (cfg.INPUT.BATCH_SIZE_TRAIN / 3)  # adapt the lr linearly
        self.bb = ['resnet50', 'pvtv2'][0]
        self.pvt_weights = [
            '/root/autodl-tmp/weights/pvt_v2_b2.pth',
            '/root/autodl-tmp/weights/pvt_v2_b1.pth',
            '/root/autodl-tmp/weights/pvt_v2_b0.pth',
            '',
        ][0]
        self.freeze_bb = False
        if 'resnet' in self.bb:
            self.bb_out_channels = [1024, 2048]
        elif 'pvt' in self.bb:
            if 'b2' in self.pvt_weights:
                self.bb_out_channels = [320, 512]
            if 'b1' in self.pvt_weights:
                self.bb_out_channels = [320, 512]
            if 'b0' in self.pvt_weights:
                self.bb_out_channels = [160, 256]
        else:
            self.bb_out_channels = [512, 1024]
        self.cxt_scene_len = self.bb_out_channels[0] * int(self.cxt_scene_enabled)     # feat-res4
        self.cxt_group_len = self.bb_out_channels[1] * int(self.cxt_group_enabled)     # feat-res5

        self.ignore_det_last_epochs = False
        self.nae_mix_res3 = True
        self.nae_multi = True
        self.nae_feature_seperate = True
        self.bn_feature_seperately = False
        if (self.nae_mix_res3 or self.nae_multi) and self.nae_feature_seperate:
            self.nae_dims = [256, 128]
        else:
            self.nae_dims = [256]
        if self.nae_multi:
            self.nae_mix_res3 = False
            if self.nae_feature_seperate:
                if self.cxt_scene_enabled:
                    self.nae_dims.append(128)
                if self.cxt_group_enabled:
                    self.nae_dims.append(128)
                self.bn_feature_seperately = False

        # Fusion on features (closed)
        self.relu_after_mlp = False
        self.bnRelu_after_conv = False
        self.fusion_attention = [0, 'sea'][0] if self.cxt_scene_enabled or self.cxt_group_enabled else 0
        self.fusion_style = {
            'mlp': 0,
            'conv': 0,
        }
        self.use_fusion = sum(self.fusion_style.values()) and not self.nae_multi
        self.feat_cxt_reid_len = 2048 + self.cxt_scene_len + self.cxt_group_len

        self.nae_norm2_len = 2048 + (self.cxt_scene_len + self.cxt_group_len) * (not bool(self.use_fusion)) * (not self.nae_multi)


class ConfigMVN():
    def __init__(self) -> None:
        self.gallery_size = 2000
        self.train_appN = 10

