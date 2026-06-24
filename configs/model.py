import os


class ConfigModel:
    def __init__(self, base) -> None:
        # Backbone settings
        self.bb = [
            'resnet50', 'resnet101',                                    # ResNet
            'swin_v1_l', 'swin_v1_b',                                   # Swin Transformer
            'dino_v3_b', 'dino_v3_l', 'dino_v3_h_plus', 'dino_v3_7b',   # DINOv3
            'pvt_v2_b2', 'pvt_v2_b1',                                   # PVT v2
        ][4]
        # Default-derived from backbone, set explicitly so it can be overridden independently of backbone name.
        self.freeze_bb_default = any(x in self.bb for x in ['dino_v3', 'swin_v1'])
        self.freeze_bb_override = None  # None uses default; set True/False to override
        self.freeze_bb = self.freeze_bb_default if self.freeze_bb_override is None else self.freeze_bb_override
        # Normalization choice per module group (default 'BN' = baseline; 'none' = no norm).
        # norm_context: 2D conv context heads (per-image, small batch) -> GN recommended (batch-independent, B=1-safe).
        # norm_nae:     NAE projector (RoI batch) -> BN baseline; LN if experimenting (1D).
        # norm_head:    BBoxRegressor / Fuser / _bn modules -> BN baseline; LN if experimenting (works 1D & 2D).
        # norm_res4:    res4_conv head fabricating feat_res4 from RoI-pooled feat_res3 (2D conv) -> BN baseline.
        # The NAE rescaler is hard-pinned to BN in code (size-1 dim makes LN/GN degenerate).
        self.norm_context = ['BN', 'GN', 'none'][0]
        self.norm_nae = ['BN', 'LN', 'none'][0]
        self.norm_head = ['BN', 'LN', 'none'][0]
        self.norm_res4 = ['BN', 'GN', 'none'][0]

        # Backbone output channels: [res3_channels, res4_channels]
        # For Swin: res3=stage2 (4*C), res4=stage3 (8*C) to match ResNet stride 16/32
        # For DINOv3: all stages have same channels (ViT architecture)
        self.bb_out_channels_collection = {
            'resnet50': [1024, 2048], 'resnet101': [1024, 2048],
            'pvt_v2_b2': [320, 512], 'pvt_v2_b1': [320, 512],
            'swin_v1_l': [768, 1536], 'swin_v1_b': [512, 1024],
            'dino_v3_7b': [4096, 4096], 'dino_v3_h_plus': [1280, 1280], 'dino_v3_l': [1024, 1024],
            'dino_v3_b': [768, 768],
        }
        self.bb_out_channels = self.bb_out_channels_collection.get(self.bb, [1024, 2048])

        # Backbone weights
        self.weights_root_dir = os.path.join(base.sys_home_dir, 'weights/cv')
        model_name_to_weights_file = {
            'pvt_v2_b2': 'pvt_v2_b2.pth', 'pvt_v2_b1': 'pvt_v2_b1.pth',
            'swin_v1_l': 'swin_large_patch4_window12_384_22kto1k.pth', 'swin_v1_b': 'swin_base_patch4_window12_384_22kto1k.pth',
            'dino_v3_7b': 'vit_7b_patch16_dinov3.lvd1689m.pth', 'dino_v3_h_plus': 'vit_huge_plus_patch16_dinov3.lvd1689m.pth',
            'dino_v3_l': 'vit_large_patch16_dinov3.lvd1689m.pth', 'dino_v3_b': 'vit_base_patch16_dinov3.lvd1689m.pth',
        }
        self.weights = {}
        for model_name, weights_file in model_name_to_weights_file.items():
            if 'dino_v3' in model_name:
                model_name_dir = 'DINOv3-timm'
            elif 'swin_v1' in model_name:
                model_name_dir = 'swin'
            elif 'pvt_v2' in model_name:
                model_name_dir = 'pvt'
            else:
                model_name_dir = ''
            self.weights[model_name] = os.path.join(self.weights_root_dir, model_name_dir, weights_file)

        # Context extraction settings
        self.cxt_scene_enabled = True
        self.cxt_group_enabled = True
        self.cxt = self.cxt_scene_enabled or self.cxt_group_enabled

        # Context extractor types: 0=AdaptiveMaxPool, 1=Attentioned(ContextExtractor1), 3=MultiScale(ContextExtractor3_group, group only)
        self.cxt_ext_scene = [0, 1][1]
        self.cxt_ext_group = [0, 1, 3][2]

        # Context feature dimensions
        self.cxt_scene_len = self.bb_out_channels[0] * int(self.cxt_scene_enabled)
        self.cxt_group_len = self.bb_out_channels[1] * int(self.cxt_group_enabled)

        # Memory optimization settings
        self.gradient_checkpointing = False  # Save ~30-40% memory, slightly slower
        self.SDPA_enabled = True  # Use PyTorch scaled dot product attention

        # Multi-part matching settings
        self.multi_part_matching = True
        # MultiPartSpliter splits RoI features into 1+2+3 horizontal stripes across 3 granularity
        # blocks (scales [1, 2, 3] in models/modules/parts.py) -> 6 part embeddings.
        self.mps_num_parts = 1 + 2 + 3
        self.mps_channels = [None, 256][0]
        self.mps_norm_len = [384, 192, 96][1] // self.mps_num_parts
        self.mps_blk = ['BasicDecBlk', 'resnet50_layer4'][1]

        # SeqHead and NAE settings
        self.nae_mix_res3 = True
        self.nae_multi = True
        self.nae_feature_seperate = True

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
        # Post-box NMS uses a higher threshold than proposal NMS: refined boxes have higher
        # mutual IoU, so a looser threshold is needed (see glcnet.py postprocess_boxes).
        self.roi_head_post_box_nms_thresh_test = 0.5
        # Minimum box edge length (px) to keep after post-box decoding (remove_small_boxes).
        self.roi_head_post_box_min_size = 0.1
        self.roi_head_detections_per_image_test = 300
        # RoI-align pooling geometry (consumed by box_roi_pool in glcnet.py).
        self.roi_pool_output_size = 14
        self.roi_pool_sampling_ratio = 2

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
