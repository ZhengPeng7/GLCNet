import os


class ConfigModel:
    def __init__(self, base) -> None:
        # Backbone settings
        self.bb = ['resnet50', 'resnet101', 'pvtv2'][0]
        self.freeze_bb = False
        self.use_bn = True

        # Backbone output channels (auto-configured based on backbone)
        self.bb_out_channels = self._get_bb_out_channels()

        # Backbone weights for PVT
        model_name_dir = 'pvt'
        self.weights_pvt = [
            os.path.join(base.sys_home_dir, 'weights/cv', model_name_dir, 'pvt_v2_b2.pth'),
            os.path.join(base.sys_home_dir, 'weights/cv', model_name_dir, 'pvt_v2_b1.pth'),
            os.path.join(base.sys_home_dir, 'weights/cv', model_name_dir, 'pvt_v2_b0.pth'),
            '',
        ][0]

        # Context extraction settings
        self.cxt_scene_enabled = True
        self.cxt_group_enabled = True
        self.cxt_group_labelledOnly = False
        self.cxt = self.cxt_scene_enabled or self.cxt_group_enabled

        # Context extractor types: 0=AdaptiveMaxPool, 1=Attentioned, 2=CoAtt, 3=MultiScale
        self.cxt_ext_scene = [0, 1, 2, 3, 4][1]
        self.cxt_ext_group = [0, 1, 2, 3, 4][3]

        # Context feature dimensions
        self.cxt_scene_len = self.bb_out_channels[0] * int(self.cxt_scene_enabled)
        self.cxt_group_len = self.bb_out_channels[1] * int(self.cxt_group_enabled)

        # Memory optimization settings
        self.gradient_checkpointing = False  # Save ~30-40% memory, slightly slower

        # Multi-part matching settings
        self.multi_part_matching = True
        self.mps_channels = [None, 256][0]
        self.mps_norm_len = [384, 192, 96][1] // (1 + 2 + 3)
        self.mps_blk = ['BasicDecBlk', 'resnet50_layer4'][1]

        # SeqHead and NAE settings
        self.seq_head = True
        self.nae_mix_res3 = True
        self.nae_multi = True
        self.nae_feature_seperate = True
        self.bn_feature_seperately = False

        # NAE dimensions (auto-configured)
        self.nae_dims = self._get_nae_dims()

        # Fusion settings (inactive by default)
        self.relu_after_mlp = False
        self.bnRelu_after_conv = False
        self.fusion_attention = [0, 'sea'][0] if self.cxt else 0
        self.fusion_style = {'mlp': 0, 'conv': 0}
        self.use_fusion = sum(self.fusion_style.values()) and not self.nae_multi

        # Derived feature dimensions
        self.feat_cxt_reid_len = self.bb_out_channels[1] + self.cxt_scene_len + self.cxt_group_len
        self.nae_norm2_len = self.bb_out_channels[1] + (self.cxt_scene_len + self.cxt_group_len) * (not bool(self.use_fusion)) * (not self.nae_multi)

        # RPN settings
        self.rpn_nms_thresh = 0.7
        self.rpn_batch_size_train = 256
        self.rpn_pos_frac_train = 0.5
        self.rpn_pos_thresh_train = 0.7
        self.rpn_neg_thresh_train = 0.3
        self.rpn_pre_nms_topn_train = 12000
        self.rpn_pre_nms_topn_test = 6000
        self.rpn_post_nms_topn_train = 2000
        self.rpn_post_nms_topn_test = 300

        # ROI Head settings
        self.roi_head_bn_neck = True
        self.roi_head_batch_size_train = 128
        self.roi_head_pos_frac_train = 0.5
        self.roi_head_pos_thresh_train = 0.5
        self.roi_head_neg_thresh_train = 0.5
        self.roi_head_score_thresh_test = 0.5
        self.roi_head_nms_thresh_test = 0.4
        self.roi_head_detections_per_image_test = 300

    def _get_bb_out_channels(self):
        if 'resnet' in self.bb:
            return [1024, 2048]
        elif 'pvt' in self.bb:
            if hasattr(self, 'weights_pvt'):
                if '_b2' in self.weights_pvt or '_b1' in self.weights_pvt:
                    return [320, 512]
                elif '_b0' in self.weights_pvt:
                    return [160, 256]
            return [320, 512]
        else:
            return [512, 1024]

    def _get_nae_dims(self):
        if (self.nae_mix_res3 or self.nae_multi) and self.nae_feature_seperate:
            dims = [256, 128]
        else:
            dims = [256]

        if self.nae_multi:
            if self.nae_feature_seperate:
                if self.cxt_scene_enabled:
                    dims.append(128)
                if self.cxt_group_enabled:
                    dims.append(128)
        return dims
