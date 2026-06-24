from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.checkpoint import checkpoint
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from losses.oim import OIMLoss
from losses.det import detection_losses
from configs import config
from models.modules import (
    SEAttention,
    MultiPartSpliter,
    ContextExtractor1,
    ContextExtractor3_group,
    build_norm_layer_1d,
)
from models.backbones import build_backbone


def _apply_fuser(fuser_att, fuser_mlp, fuser_conv, feats):
    if fuser_att:
        feats["feat_res4"] = fuser_att(feats["feat_res4"])
    if fuser_mlp:
        feats["feat_res4"] = fuser_mlp(feats["feat_res4"].squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
    elif fuser_conv:
        feats["feat_res4"] = fuser_conv(feats["feat_res4"].squeeze(-1)).unsqueeze(-1)
    return feats


class GLCNet(nn.Module):
    def __init__(self):
        super(GLCNet, self).__init__()
        print('Using backbone:', config.bb)
        assert config.bb in config.bb_out_channels_collection, f'Unsupported backbone: {config.bb}'
        backbone, box_head = build_backbone(config.bb, pretrained=True)

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=config.rpn_pre_nms_topn_train, testing=config.rpn_pre_nms_topn_test
        )
        post_nms_top_n = dict(
            training=config.rpn_post_nms_topn_train, testing=config.rpn_post_nms_topn_test
        )
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=config.rpn_pos_thresh_train,
            bg_iou_thresh=config.rpn_neg_thresh_train,
            batch_size_per_image=config.rpn_batch_size_train,
            positive_fraction=config.rpn_pos_frac_train,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=config.rpn_nms_thresh,
        )

        faster_rcnn_predictor = FastRCNNPredictor(box_head.out_channels[1], 2)
        reid_head = deepcopy(box_head)
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res3"], output_size=config.roi_pool_output_size, sampling_ratio=config.roi_pool_sampling_ratio
        )
        box_predictor = BBoxRegressor(box_head.out_channels[1], num_classes=2, bn_neck=config.roi_head_bn_neck)

        fusion_layers_att = []
        fusion_layers_mlp = []
        fusion_layers_conv = []
        if config.cxt:
            if config.cxt_scene_enabled:
                if config.cxt_ext_scene == 1:
                    self.cxt_scene_extractor = ContextExtractor1(config.cxt_scene_len)
                else:
                    self.cxt_scene_extractor = nn.Sequential(
                        nn.AdaptiveMaxPool2d(1)
                    )
            if config.cxt_group_enabled:
                if config.cxt_ext_group == 1:
                    self.cxt_group_extractor = ContextExtractor1(config.cxt_group_len)
                elif config.cxt_ext_group == 3:
                    self.cxt_group_extractor = ContextExtractor3_group(config.cxt_group_len)
                else:
                    self.cxt_group_extractor = nn.Sequential(
                        nn.AdaptiveMaxPool2d(1)
                    )

            if config.use_fusion:
                if config.fusion_attention == 'sea':
                    fusion_layers_att.append(
                        SEAttention(config.feat_cxt_reid_len)
                    )
                channel_opt = config.bb_out_channels[1]
                for k, v in config.fusion_style.items():
                    for idx_layer in range(v):
                        if not idx_layer:
                            channel_ipt = config.feat_cxt_reid_len
                        else:
                            channel_ipt = channel_opt
                        if k == 'mlp':
                            fusion_layers_mlp.append(nn.Linear(channel_ipt, channel_opt))
                            if config.relu_after_mlp:
                                fusion_layers_mlp.append(nn.ReLU())
                        elif k == 'conv':
                            fusion_layers_conv.append(nn.Conv1d(channel_ipt, channel_opt, 1, 1))
                            if config.bnRelu_after_conv:
                                # Use eps=1e-3 for bf16 compatibility
                                fusion_layers_conv.append(build_norm_layer_1d(channel_opt, config.norm_head, eps=1e-3))
                                fusion_layers_conv.append(nn.ReLU())
        if config.multi_part_matching:
            self.mps = MultiPartSpliter(out_channels=config.mps_channels)
        self.fuser_att = nn.Sequential(*fusion_layers_att)
        self.fuser_mlp = nn.Sequential(*fusion_layers_mlp)
        self.fuser_conv = nn.Sequential(*fusion_layers_conv)
        self.fuser = [self.fuser_att, self.fuser_mlp, self.fuser_conv]
        roi_heads = SeqRoIHeads(
            # OIM
            num_pids=config.lut_size,
            num_cq_size=config.cq_size,
            oim_momentum=config.oim_momentum,
            oim_scalar=config.oim_scalar,
            # GLCNet
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=config.roi_head_pos_thresh_train,
            bg_iou_thresh=config.roi_head_neg_thresh_train,
            batch_size_per_image=config.roi_head_batch_size_train,
            positive_fraction=config.roi_head_pos_frac_train,
            bbox_reg_weights=None,
            score_thresh=config.roi_head_score_thresh_test,
            nms_thresh=config.roi_head_nms_thresh_test,
            detections_per_img=config.roi_head_detections_per_image_test,
            cxt_scene_extractor=self.cxt_scene_extractor if config.cxt_scene_enabled else None,
            cxt_group_extractor=self.cxt_group_extractor if config.cxt_group_enabled else None,
            fuser=self.fuser if config.use_fusion else [[], [], []],
            mps=self.mps if config.multi_part_matching else None
        )

        transform = GeneralizedRCNNTransform(
            min_size=config.input_min_size,
            max_size=config.input_max_size,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform

        # loss weights
        self.lw_rpn_reg = config.lw_rpn_reg
        self.lw_rpn_cls = config.lw_rpn_cls
        self.lw_proposal_reg = config.lw_proposal_reg
        self.lw_proposal_cls = config.lw_proposal_cls
        self.lw_box_reg = config.lw_box_reg
        self.lw_box_cls = config.lw_box_cls
        self.lw_box_reid = config.lw_box_reid

    def inference(self, images, targets=None, query_img_as_gallery=False):
        """
        query_img_as_gallery: Set to True to detect all people in the query image.
            Meanwhile, the gt box should be the first of the detected boxes.
            This option serves CBGM.
        """
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)  # {feat_res3}; feat_res4 is fabricated by the Res4Head
        rpn_features = features

        if query_img_as_gallery:
            assert targets is not None

        proposals, _ = self.rpn(images, rpn_features, targets)
        detections, _ = self.roi_heads(
            features, proposals, images.image_sizes, targets, query_img_as_gallery
        )

        if targets is not None and not query_img_as_gallery:
            # with targets, inference on query boxes only.
            # query
            boxes = [t["boxes"] for t in targets]
            box_features = self.roi_heads.box_roi_pool(features, boxes, images.image_sizes)
            if config.multi_part_matching:
                mps_feat_dict = self.mps(box_features)
            box_features = self.roi_heads.reid_head(box_features)
            if config.multi_part_matching:
                box_features.update(mps_feat_dict)
            if config.cxt:
                # Re-inject the image-level context (cxt_scene/cxt_group) computed during the
                # detection forward, expanded to the query boxes; then fuse. See
                # SeqRoIHeads._inject_precomputed_context (single source of truth with the gt branch).
                box_features = self.roi_heads._inject_precomputed_context(box_features)
                if not config.nae_multi:
                    box_features = _apply_fuser(self.fuser_att, self.fuser_mlp, self.fuser_conv, box_features)
            embeddings, _ = self.roi_heads.embedding_head(box_features if config.nae_mix_res3 or config.nae_multi else OrderedDict([['feat_res4', box_features["feat_res4"]]]))
            return embeddings.split(1, 0)
        else:
            # if targets is None:   1. without targets, extracting gallery.
            # else:                 2. with targets, inference on query and its surrounding persons.
            # gallery
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            return detections

    def forward(self, images, targets=None, query_img_as_gallery=False):
        if not self.training:
            return self.inference(images, targets, query_img_as_gallery)

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)  # {feat_res3}; feat_res4 is fabricated by the Res4Head
        proposals, proposal_losses = self.rpn(images, features, targets)
        _, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # rename rpn losses to be consistent with detection losses
        proposal_losses["loss_rpn_reg"] = proposal_losses.pop("loss_rpn_box_reg")
        proposal_losses["loss_rpn_cls"] = proposal_losses.pop("loss_objectness")

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # apply loss weights - keep ALL losses in float32 for numerical stability
        # DO NOT convert back to bf16, as this causes gradient quantization
        losses["loss_rpn_reg"] = losses["loss_rpn_reg"].float() * self.lw_rpn_reg
        losses["loss_rpn_cls"] = losses["loss_rpn_cls"].float() * self.lw_rpn_cls
        losses["loss_proposal_reg"] = losses["loss_proposal_reg"].float() * self.lw_proposal_reg
        losses["loss_proposal_cls"] = losses["loss_proposal_cls"].float() * self.lw_proposal_cls
        losses["loss_box_reg"] = losses["loss_box_reg"].float() * self.lw_box_reg
        losses["loss_box_cls"] = losses["loss_box_cls"].float() * self.lw_box_cls
        losses["loss_box_reid"] = losses["loss_box_reid"].float() * self.lw_box_reid
        return losses


class SeqRoIHeads(RoIHeads):
    def __init__(
        self,
        num_pids,
        num_cq_size,
        oim_momentum,
        oim_scalar,
        faster_rcnn_predictor,
        reid_head,
        cxt_scene_extractor,
        cxt_group_extractor,
        fuser,
        mps,
        *args,
        **kwargs
    ):
        super(SeqRoIHeads, self).__init__(*args, **kwargs)
        if not config.nae_multi:
            if config.nae_mix_res3:
                featmap_names = ['feat_res3', 'feat_res4']
                in_channels=[config.bb_out_channels[0], config.nae_norm2_len]
            else:
                featmap_names = ['feat_res4']
                in_channels=[config.nae_norm2_len]
        else:
            featmap_names = ['feat_res3', 'feat_res4']
            in_channels=[reid_head.out_channels[0], reid_head.out_channels[1]]
            if config.cxt_scene_enabled:
                featmap_names.append('cxt_scene')
                in_channels.append(config.cxt_scene_len)
            if config.cxt_group_enabled:
                featmap_names.append('cxt_group')
                in_channels.append(config.cxt_group_len)
        if config.multi_part_matching:
            self.mps = mps
            # When using resnet50_layer4 as MPS block, output is always 2048 regardless of backbone
            if config.mps_blk == 'resnet50_layer4':
                mps_channels = config.mps_channels if config.mps_channels else 2048
            else:
                mps_channels = config.mps_channels if config.mps_channels else config.bb_out_channels[1]
            feat_names_channels = {
                'feat_granularity_1': mps_channels,
                'feat_granularity_2_horizon_1': mps_channels, 'feat_granularity_2_horizon_2': mps_channels,
                'feat_granularity_3_horizon_1': mps_channels, 'feat_granularity_3_horizon_2': mps_channels, 'feat_granularity_3_horizon_3': mps_channels,
            }
            for feat_name, feat_channels in feat_names_channels.items():
                featmap_names.append(feat_name)
                in_channels.append(feat_channels)
        self.embedding_head = NormAwareEmbedding(
            featmap_names=featmap_names,
            in_channels=in_channels,
            dim=config.nae_dims,
        )
        self.reid_loss = OIMLoss(self.embedding_head.dim_total, num_pids, num_cq_size, oim_momentum, oim_scalar)
        self.faster_rcnn_predictor = faster_rcnn_predictor
        self.reid_head = reid_head
        # rename the method inherited from parent class
        self.postprocess_proposals = self.postprocess_detections
        self.cxt_scene_extractor = cxt_scene_extractor
        self.cxt_group_extractor = cxt_group_extractor
        self.fuser_att, self.fuser_mlp, self.fuser_conv = fuser

    def _inject_precomputed_context(self, box_features):
        """Re-inject the cached image-level context (cxt_scene/cxt_group from x_embedding_head,
        set during the detection forward) into box_features, expanded to its box count. Eval-only;
        single source of truth shared by inference() query and the query_img_as_gallery gt branch.
        The caller applies the fuser afterwards (its fuser reference differs by call site).

        NOTE: relies on batch_size_test==1 (hard-asserted at eval.py:124). The per-image context is
        stored in x_embedding_head expanded over that image's detection boxes, so mean(dim=0) below
        exactly recovers the single image-level context. With a multi-image eval batch (unsupported)
        the mean would blend images; do not enable batch_size_test>1 without a per-image rewrite."""
        num_boxes = box_features['feat_res4'].shape[0]
        orig_dtype = box_features['feat_res4'].dtype
        for key, enabled in (('cxt_scene', config.cxt_scene_enabled), ('cxt_group', config.cxt_group_enabled)):
            if not enabled:
                continue
            # mean(dim=0) collapses the per-box-expanded storage back to the (batch=1) image-level context
            ctx = torch.mean(self.x_embedding_head[key].float(), dim=0).unsqueeze(0).expand(num_boxes, -1, -1, -1).to(orig_dtype)
            if config.nae_multi:
                box_features[key] = ctx
            else:
                box_features["feat_res4"] = torch.cat([box_features["feat_res4"], ctx], 1)
        return box_features

    def forward(self, features, proposals, image_shapes, targets=None, query_img_as_gallery=False):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if self.training:
            proposals, _, proposal_pid_labels, proposal_reg_targets = self.select_training_samples(
                proposals, targets
            )

        # ------------------- Faster R-CNN head ------------------ #
        proposal_features = self.box_roi_pool(features, proposals, image_shapes)
        proposal_features = self.box_head(proposal_features)
        proposal_cls_scores, proposal_regs = self.faster_rcnn_predictor(
            proposal_features["feat_res4"]
        )

        if self.training:
            boxes = self.get_boxes(proposal_regs, proposals, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            boxes, _, box_pid_labels, box_reg_targets = self.select_training_samples(boxes, targets)
        else:
            # invoke the postprocess method inherited from parent class to process proposals
            boxes, scores, box_pid_labels = self.postprocess_proposals(
                proposal_cls_scores, proposal_regs, proposals, image_shapes
            )
            # 0-detection fallback: the detector found no person but a query box is provided
            # (query / query-as-gallery / use_gt eval). Treat that box as the sole detection so
            # context is built like a single-person image in training (cxt_scene = image-level,
            # cxt_group = the box itself) instead of crashing downstream. The box is given a 0
            # score so it is filtered out of the detection output (postprocess_boxes), leaving the
            # force-included GT box as the sole result; x_embedding_head is set earlier and is
            # unaffected. Pure gallery (targets=None) and the normal (>=1 detection) path are
            # untouched -> bit-identical.
            if targets is not None and boxes[0].shape[0] == 0:
                boxes = [t["boxes"] for t in targets]
                scores = [torch.zeros(b.shape[0], device=b.device) for b in boxes]
                box_pid_labels = [torch.ones(b.shape[0], dtype=torch.long, device=b.device) for b in boxes]
        if config.cxt_group_enabled:
            whether_box_has_person_lst_per_image = []
            for box_pid_label in box_pid_labels:
                whether_box_has_person_lst = box_pid_label > 0
                whether_box_has_person_lst_per_image.append(whether_box_has_person_lst)

        cws = True
        gt_det = None

        if boxes[0].shape[0] > 0:
            # --------------------- Baseline head -------------------- #
            # Basic forward process and context extraction.
            box_features = self.box_roi_pool(features, boxes, image_shapes)
            if config.multi_part_matching:
                # features.shape == batch_size, 1024, 58, 94
                if self.training and config.gradient_checkpointing:
                    mps_feat_dict = checkpoint(self.mps, box_features, use_reentrant=False)
                else:
                    mps_feat_dict = self.mps(box_features)
            if self.training and config.gradient_checkpointing:
                box_features = checkpoint(self.reid_head, box_features, use_reentrant=False)
            else:
                box_features = self.reid_head(box_features)
            if config.multi_part_matching:
                box_features.update(mps_feat_dict)
            box_regs = self.box_predictor(box_features["feat_res4"])
            if config.cxt:
                feat_res4_ori = box_features['feat_res4']
                if config.cxt_scene_enabled:
                    if self.training and config.gradient_checkpointing:
                        cxt_scene = checkpoint(self.cxt_scene_extractor, features['feat_res3'], use_reentrant=False)
                    else:
                        cxt_scene = self.cxt_scene_extractor(features['feat_res3'])

                    cxt_scene_lst = []
                    for idx_bs, box in enumerate(boxes):
                        # I repeat the scene feature for box times to add the scene context to each box -- Equip all objects in this image with the scene context here.
                        cxt_scene_lst.append(cxt_scene[idx_bs:idx_bs+1].expand(box.shape[0], -1, -1, -1))
                    cxt_scene_proposalNum = torch.cat(cxt_scene_lst, dim=0)

                    if config.nae_multi:
                        box_features['cxt_scene'] = cxt_scene_proposalNum
                    else:
                        box_features["feat_res4"] = torch.cat([box_features["feat_res4"], cxt_scene_proposalNum.to(feat_res4_ori.dtype)], 1)
                if config.cxt_group_enabled:
                    # We collect the features of detected boxes in each image, which are used as the group context to the persons in this image.
                    feat_res4_per_image = []
                    num_box = 0
                    for box in boxes:
                        # Divide the features in a batch into features in each image.
                        feat_res4_per_image.append(feat_res4_ori[num_box:num_box+box.shape[0]])
                        num_box += box.shape[0]
                    cxt_group_per_image = []
                    for feat_res4, whether_box_has_person_lst in zip(feat_res4_per_image, whether_box_has_person_lst_per_image):
                        # Select the features of boxes which have persons. Each image has 128 boxes.
                        feat_persons = feat_res4[whether_box_has_person_lst]
                        if feat_persons.shape[0] == 0:
                            # No person boxes in this image -> use all boxes as group-context proxy (avoid NaN from empty mean).
                            feat_persons = feat_res4
                        # mean aggregation computed in fp32 for numerical stability (not the whole context path)
                        feat_persons_squeezed = feat_persons.mean(dim=0, dtype=torch.float32)
                        feat_persons_squeezed = feat_persons_squeezed.unsqueeze(0)
                        cxt_group_per_image.append(feat_persons_squeezed)
                    cxt_group_per_image = torch.cat(cxt_group_per_image, dim=0)
                    if self.training and config.gradient_checkpointing:
                        cxt_group_per_image = checkpoint(self.cxt_group_extractor, cxt_group_per_image, use_reentrant=False)
                    else:
                        cxt_group_per_image = self.cxt_group_extractor(cxt_group_per_image)

                    cxt_group_lst = []
                    for idx_bs, box in enumerate(boxes):
                        cxt_group_lst.append(cxt_group_per_image[idx_bs:idx_bs+1].expand(box.shape[0], -1, -1, -1))
                    cxt_group_proposalNum = torch.cat(cxt_group_lst, dim=0)

                    if config.nae_multi:
                        box_features['cxt_group'] = cxt_group_proposalNum.to(feat_res4_ori.dtype)
                    else:
                        box_features["feat_res4"] = torch.cat([box_features["feat_res4"], cxt_group_proposalNum.to(feat_res4_ori.dtype)], 1)
                if not config.nae_multi:
                    box_features = _apply_fuser(self.fuser_att, self.fuser_mlp, self.fuser_conv, box_features)
            if config.nae_mix_res3 or config.nae_multi:
                x_embedding_head = box_features
            else:
                x_embedding_head = OrderedDict([['feat_res4', box_features["feat_res4"]]])
            self.x_embedding_head = x_embedding_head

            if not self.training and query_img_as_gallery:
                # group context of query: When regarding the query image as gallery, GT boxes may be excluded from detected boxes. To avoid this, we compulsorily include GT in the detection results.
                # Additionally, CWS should be disabled as the confidences of these people in query image are 1.
                cws = False
                gt_boxes = [targets[0]["boxes"]]
                gt_box_features = self.box_roi_pool(features, gt_boxes, image_shapes)
                if config.multi_part_matching:
                    mps_feat_dict = self.mps(gt_box_features)
                gt_box_features = self.reid_head(gt_box_features)
                if config.multi_part_matching:
                    gt_box_features.update(mps_feat_dict)
                if config.cxt:
                    # Re-inject the image-level context, expanded to the gt boxes (single source
                    # of truth with inference()'s query path; see _inject_precomputed_context).
                    gt_box_features = self._inject_precomputed_context(gt_box_features)
                if not config.nae_multi:
                    gt_box_features = _apply_fuser(self.fuser_att, self.fuser_mlp, self.fuser_conv, gt_box_features)
                embeddings, _ = self.embedding_head(gt_box_features if config.nae_mix_res3 or config.nae_multi else OrderedDict([['feat_res4', gt_box_features["feat_res4"]]]))
                gt_det = {"boxes": targets[0]["boxes"], "embeddings": embeddings}

            box_embeddings, box_cls_scores = self.embedding_head(x_embedding_head)
        else:
            # 0 detections. The 0-detection fallback (after postprocess_proposals) fills boxes from
            # targets whenever targets is provided, so this branch is reached only by a pure gallery
            # image (targets is None) with no person detected -> return an empty result.
            # (eval-only; training never hits boxes==0 due to fixed-size sampling.)
            assert not self.training
            return [dict(boxes=torch.zeros(0, 4), labels=torch.zeros(0), scores=torch.zeros(0),
                         embeddings=torch.zeros(0, self.embedding_head.dim_total))], []

        result, losses = [], {}
        if self.training:
            proposal_labels = [y.clamp(0, 1) for y in proposal_pid_labels]
            box_labels = [y.clamp(0, 1) for y in box_pid_labels]
            losses = detection_losses(
                proposal_cls_scores,
                proposal_regs,
                proposal_labels,
                proposal_reg_targets,
                box_cls_scores,
                box_regs,
                box_labels,
                box_reg_targets,
            )
            loss_box_reid = self.reid_loss(box_embeddings, box_pid_labels)
            losses.update(loss_box_reid=loss_box_reid)
            # the return value of cross_entropy() in pytorch was changed from `0.0 to nan` when target.numel() == 0.
            # reference: https://github.com/pytorch/pytorch/issues/50224
        else:
            # The IoUs of these boxes are higher than that of proposals,
            # so a higher NMS threshold is needed
            orig_thresh = self.nms_thresh
            self.nms_thresh = config.roi_head_post_box_nms_thresh_test
            boxes, scores, embeddings, labels = self.postprocess_boxes(
                box_cls_scores,
                box_regs,
                box_embeddings,
                boxes,
                image_shapes,
                fcs=scores,
                gt_det=gt_det,
                cws=cws,
            )
            # set to original thresh after finishing postprocess
            self.nms_thresh = orig_thresh
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i], labels=labels[i], scores=scores[i], embeddings=embeddings[i]
                    )
                )
        return result, losses

    def get_boxes(self, box_regression, proposals, image_shapes):
        """
        Get boxes from proposals.
        """
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_boxes = pred_boxes.split(boxes_per_image, 0)

        all_boxes = []
        for boxes, image_shape in zip(pred_boxes, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # remove predictions with the background label
            boxes = boxes[:, 1:].reshape(-1, 4)
            all_boxes.append(boxes)

        return all_boxes

    def postprocess_boxes(
        self,
        class_logits,
        box_regression,
        embeddings,
        proposals,
        image_shapes,
        fcs=None,
        gt_det=None,
        cws=True,
    ):
        """
        Similar to RoIHeads.postprocess_detections, but can handle embeddings and implement
        First Classification Score (FCS).
        """
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        if fcs is not None:
            # Fist Classification Score (FCS) — concat per-image scores to align with the full
            # embeddings/pred_boxes batch (equivalent to fcs[0] when batch_size_test == 1).
            pred_scores = torch.cat(fcs)
        else:
            # Compute sigmoid in float32 for numerical stability with bf16/fp16
            # During eval, keep float32 for precision; during training, convert back to save memory
            pred_scores = torch.sigmoid(class_logits.float())
            if self.training:
                pred_scores = pred_scores.to(class_logits.dtype)
        if cws:
            # Confidence Weighted Similarity (CWS) - compute in float32 for numerical stability
            # During eval, embeddings are already float32 from NormAwareEmbedding
            embeddings = embeddings.float() * pred_scores.view(-1, 1).float()
            if self.training:
                embeddings = embeddings.to(class_logits.dtype)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(
            pred_boxes, pred_scores, pred_embeddings, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.ones(scores.size(0), device=device)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head.dim_total)

            # remove low scoring boxes (compare in float32 for bf16/fp16 precision)
            inds = torch.nonzero(scores.float() > float(self.score_thresh)).squeeze(1)
            boxes, scores, labels, embeddings = (
                boxes[inds],
                scores[inds],
                labels[inds],
                embeddings[inds],
            )

            # remove empty boxes (use larger min_size for bf16 numerical stability)
            keep = box_ops.remove_small_boxes(boxes.float(), min_size=config.roi_head_post_box_min_size)
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            if gt_det is not None:
                # include GT into the detection results, PREPENDED so the GT/query box has the lowest
                # input index: batched_nms breaks score ties by input order, so a detection whose score
                # saturates to 1.0 (bf16) and overlaps the GT cannot suppress it. Keeps the GT/query box
                # at index 0 (downstream GT-first asserts / CBGM rely on this). Non-tie case is unchanged
                # (GT score 1.0 already sorts first).
                boxes = torch.cat((gt_det["boxes"], boxes), dim=0)
                labels = torch.cat((torch.full((gt_det["boxes"].shape[0],), 1.0, dtype=torch.float32, device=device), labels), dim=0)
                scores = torch.cat((torch.full((gt_det["boxes"].shape[0],), 1.0, dtype=torch.float32, device=device), scores), dim=0)
                embeddings = torch.cat((gt_det["embeddings"], embeddings), dim=0)

            # non-maximum suppression in float32 for precise IoU calculation
            keep = box_ops.batched_nms(boxes.float(), scores.float(), labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embeddings)

        return all_boxes, all_scores, all_embeddings, all_labels


class NormAwareEmbedding(nn.Module):
    """
    Implements the Norm-Aware Embedding proposed in
    Chen, Di, et al. "Norm-aware embedding for efficient person search." CVPR 2020.
    """

    def __init__(self, featmap_names, in_channels, dim):
        super(NormAwareEmbedding, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = dim

        self.projectors = nn.ModuleDict()
        if len(dim) == 1:
            indv_dims = self._split_embedding_dim(self.dim[0])
        else:
            indv_dims = list(dim)
            if config.multi_part_matching:
                num_parts = config.mps_num_parts    # 1 + 2 + 3 parts; see MultiPartSpliter (models/modules/parts.py).
                for _ in range(num_parts):
                    indv_dims.append(config.mps_norm_len)
        self.dim_total = sum(indv_dims)

        # nae_dims is derived from 5 interacting toggles (configs/model.py _get_nae_dims); fail loud on a
        # misconfiguration here rather than letting the zip below silently truncate to the shortest and
        # mis-split the embedding (featmap_names/in_channels/indv_dims must be 1:1).
        assert len(self.featmap_names) == len(self.in_channels) == len(indv_dims), (
            f"NAE dim mismatch: featmap_names={len(self.featmap_names)}, "
            f"in_channels={len(self.in_channels)}, indv_dims={len(indv_dims)}"
        )

        # Use eps=1e-3 for bf16 compatibility (default 1e-5 is borderline)
        bn_eps = 1e-3
        for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            proj = nn.Sequential(nn.Linear(in_channel, indv_dim), build_norm_layer_1d(indv_dim, config.norm_nae, eps=bn_eps))
            init.normal_(proj[0].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            if not isinstance(proj[1], nn.Identity):  # norm_nae='none' -> Identity has no affine params
                init.normal_(proj[1].weight, std=0.01)
                init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        # rescaler hard-pinned to BN: it normalizes the size-1 L2-norm scalar (the norm-aware core);
        # LN/GN are mathematically degenerate on a size-1 dim, so this is never config-driven.
        self.rescaler = build_norm_layer_1d(1, 'BN', eps=bn_eps, affine=True)

    def forward(self, featmaps):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """

        assert len(featmaps) == len(self.featmap_names)
        # During evaluation, keep embeddings in float32 for precision; during training, convert back to save memory
        cast_back_to_orig = self.training
        if len(featmaps) == 1:
            for k, v in featmaps.items():
                pass
            v = self._flatten_fc_input(v)
            orig_dtype = v.dtype
            embeddings = self.projectors[k](v)
            # Compute norm and division in float32 for numerical stability with bf16/fp16
            norms = embeddings.float().norm(2, 1, keepdim=True)
            embeddings = embeddings.float() / norms.clamp(min=1e-4).expand_as(embeddings)
            if cast_back_to_orig:
                embeddings = embeddings.to(orig_dtype)
            norms = self.rescaler(norms.to(orig_dtype)).squeeze(1)
            return embeddings, norms
        else:
            outputs = []
            orig_dtype = None
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                if orig_dtype is None:
                    orig_dtype = v.dtype
                proj_feat = self.projectors[k](v)
                outputs.append(proj_feat)
            embeddings = torch.cat(outputs, dim=1)
            # Compute norm and division in float32 for numerical stability
            norms = embeddings.float().norm(2, 1, keepdim=True)
            embeddings = embeddings.float() / norms.clamp(min=1e-4).expand_as(embeddings)
            if cast_back_to_orig:
                embeddings = embeddings.to(orig_dtype)
            norms = self.rescaler(norms.to(orig_dtype)).squeeze(1)
            return embeddings, norms

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x

    def _split_embedding_dim(self, dim):
        parts = len(self.in_channels)
        tmp = [dim // parts] * parts
        if sum(tmp) == dim:
            return tmp
        else:
            res = dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == dim
            return tmp


class BBoxRegressor(nn.Module):
    """
    Bounding box regression layer.
    """

    def __init__(self, in_channels, num_classes=2, bn_neck=True):
        """
        Args:
            in_channels (int): Input channels.
            num_classes (int, optional): Defaults to 2 (background and pedestrian).
            bn_neck (bool, optional): Whether to use BN after Linear. Defaults to True.
        """
        super(BBoxRegressor, self).__init__()
        if bn_neck:
            # Use eps=1e-3 for bf16 compatibility
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes), build_norm_layer_1d(4 * num_classes, config.norm_head, eps=1e-3)
            )
            init.normal_(self.bbox_pred[0].weight, std=0.01)
            init.constant_(self.bbox_pred[0].bias, 0)
            if not isinstance(self.bbox_pred[1], nn.Identity):  # norm_head='none' -> Identity has no affine params
                init.normal_(self.bbox_pred[1].weight, std=0.01)
                init.constant_(self.bbox_pred[1].bias, 0)
        else:
            self.bbox_pred = nn.Linear(in_channels, 4 * num_classes)
            init.normal_(self.bbox_pred.weight, std=0.01)
            init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas
