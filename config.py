import os
from defaults import get_default_cfg


cfg = get_default_cfg()


class Config():
    def __init__(self) -> None:
        self.multi_part_matching = True    # 1.1 min for 200 steps w/ False.
        self.mps_channels = [None, 256][0]
        self.mps_norm_len = [384, 192, 96][1] // (1 + 2 + 3)
        self.mps_blk = ['BasicDecBlk', 'resnet50_layer4'][1]    # [1.3min, 1.6min] for 200 steps. 'resnet50_layer4' is not applicable for backbones other than resnet50.
        # Context Features
        self.cxt_scene_enabled = True
        self.cxt_group_enabled = True
        self.cxt_group_labelledOnly = False
        self.cxt = self.cxt_scene_enabled or self.cxt_group_enabled

        self.cxt_ext_scene = [0, 1, 2, 3, 4][1]     # [1] is the best one.
        self.cxt_ext_group = [0, 1, 2, 3, 4][3]     # [1] is the best one.
        self.lr = 0.003
        self.bb = ['resnet50', 'resnet101', 'pvtv2'][0]
        self.weights_pvt = [
            os.path.join(cfg.SYS_HOME_DIR, 'weights/pvt_v2_b2.pth'),
            os.path.join(cfg.SYS_HOME_DIR, 'weights/pvt_v2_b1.pth'),
            os.path.join(cfg.SYS_HOME_DIR, 'weights/pvt_v2_b0.pth'),
            '',
        ][0]
        self.use_bn = True  # not ('pvtv2' == self.bb and 'pvt_v2_b2.pth' in self.weights_pvt)
        self.freeze_bb = False
        if 'resnet' in self.bb:
            self.bb_out_channels = [1024, 2048]
        elif 'pvt' in self.bb:
            if '_b2' in self.weights_pvt:
                self.bb_out_channels = [320, 512]
            elif '_b1' in self.weights_pvt:
                self.bb_out_channels = [320, 512]
            elif '_b0' in self.weights_pvt:
                self.bb_out_channels = [160, 256]
        else:
            self.bb_out_channels = [512, 1024]

        self.cxt_scene_len = self.bb_out_channels[0] * int(self.cxt_scene_enabled)     # feat-res4
        self.cxt_group_len = self.bb_out_channels[1] * int(self.cxt_group_enabled)     # feat-res5

        self.seq_head = True
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
        self.train_appN = [10, 30, 50, 70, 100][0]      # Settings of amount level of training data. [10, 30, 70] are included in the official paper, since [70] already includes most images.
        self.gallery_size = [2000, 4000, 10000][0]      # Settings of gallery size used for evaluation.

