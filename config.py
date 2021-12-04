class Config():
    def __init__(self) -> None:
        # extensions
        self.co_res4 = False
        self.co_res5 = False

        self.cxt_feat = True
        self.psn_feat = False
        self.psn_feat_labelledOnly = False

        self.nae_mix_res4 = True
        self.nae_multi = True
        self.nae_feature_seperate = True
        self.bn_feature_seperately = False
        if (self.nae_mix_res4 or self.nae_multi) and self.nae_feature_seperate:
            self.nae_dims = [256, 128]
        else:
            self.nae_dims = [256]
        if self.nae_multi:
            self.nae_mix_res4 = False
            if self.nae_feature_seperate:
                if self.cxt_feat:
                    self.nae_dims.append(128)
                if self.psn_feat:
                    self.nae_dims.append(128)
                self.bn_feature_seperately = False

        self.fusion_style = {
            'mlp': 0,
            'conv': 0,
        }
        self.relu_after_mlp = False
        self.bnRelu_after_conv = False
        self.use_fusion = sum(self.fusion_style.values()) and not Config().nae_multi
        self.fusion_attention = [0, 'sea'][1] if self.cxt_feat or self.psn_feat else 0
        self.conv_before_fusion_scenario = True
        self.conv_before_fusion_psn = True

        self.cxt_feat_len = 1024 * int(self.cxt_feat)
        self.psn_feat_len = 2048 * int(self.psn_feat)
        self.feat_cxt_reid_len = 2048 + self.cxt_feat_len + self.psn_feat_len
        self.nae_norm2_len = 2048 + (self.cxt_feat_len + self.psn_feat_len) * (not bool(self.use_fusion)) * (not self.nae_multi)


class ConfigMVN():
    def __init__(self) -> None:
        self.gallery_size = 2000
        self.train_appN = 10

