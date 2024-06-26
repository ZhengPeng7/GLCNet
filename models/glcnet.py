from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from losses.oim import OIMLoss
from config import Config
from models.modules import SEAttention

from models.resnet import build_resnet50, build_resnet101
from models.pvt import build_pvt
from models.modules import MultiPartSpliter

from models.cxt_ext import ContextExtractor1, ContextExtractor2, ContextExtractor3_scene, ContextExtractor3_group


config = Config()


class GLCNet(nn.Module):
    def __init__(self, cfg):
        super(GLCNet, self).__init__()
        print('Using backbone:', config.bb)
        if config.bb == 'resnet50':
            backbone, box_head = build_resnet50(pretrained=True)
        elif config.bb == 'resnet101':
            backbone, box_head = build_resnet101(pretrained=True)
        elif config.bb == 'pvtv2':
            backbone, box_head = build_pvt(weights=config.weights_pvt)
        else:
            print('Not a valid backbone in `config.py`')
            exit()

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=cfg.MODEL.RPN.PRE_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.PRE_NMS_TOPN_TEST
        )
        post_nms_top_n = dict(
            training=cfg.MODEL.RPN.POST_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.POST_NMS_TOPN_TEST
        )
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        )

        faster_rcnn_predictor = FastRCNNPredictor(box_head.out_channels[1], 2)
        reid_head = deepcopy(box_head)
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res3"], output_size=14, sampling_ratio=2
        )
        box_predictor = BBoxRegressor(box_head.out_channels[1], num_classes=2, bn_neck=cfg.MODEL.ROI_HEAD.BN_NECK)

        fusion_layers_att = []
        fusion_layers_mlp = []
        fusion_layers_conv = []
        if config.cxt:
            if config.cxt_scene_enabled:
                if config.cxt_ext_scene == 1:
                    self.cxt_scene_extractor = ContextExtractor1(config.cxt_scene_len)
                elif config.cxt_ext_scene == 2:
                    self.cxt_scene_extractor = ContextExtractor2(config.cxt_scene_len)
                elif config.cxt_ext_scene == 3:
                    self.cxt_scene_extractor = ContextExtractor3_scene(config.cxt_scene_len)
                else:
                    self.cxt_scene_extractor = nn.Sequential(
                        nn.AdaptiveMaxPool2d(1)
                    )
            if config.cxt_group_enabled:
                if config.cxt_ext_group == 1:
                    self.cxt_group_extractor = ContextExtractor1(config.cxt_group_len)
                elif config.cxt_ext_group == 2:
                    self.cxt_group_extractor = ContextExtractor2(config.cxt_group_len)
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
                channel_opt = 2048
                for k, v in config.fusion_style.items():
                    for idx_layer in range(v):
                        if not idx_layer:
                            channel_ipt = config.feat_cxt_reid_len
                        else:
                            channel_ipt = channel_opt
                        if k == 'mlp':
                            fusion_layers_mlp.append(nn.Linear(channel_ipt, channel_opt))
                            if config.relu_after_mlp:
                                fusion_layers_mlp.append(nn.ReLU(inplace=True))
                        elif k == 'conv':
                            fusion_layers_conv.append(nn.Conv1d(channel_ipt, channel_opt, 1, 1))
                            if config.bnRelu_after_conv:
                                fusion_layers_conv.append(nn.BatchNorm1d(channel_opt))
                                fusion_layers_conv.append(nn.ReLU(inplace=True))
        if config.multi_part_matching:
            self.mps = MultiPartSpliter(out_channels=config.mps_channels)
        self.fuser_att = nn.Sequential(*fusion_layers_att)
        self.fuser_mlp = nn.Sequential(*fusion_layers_mlp)
        self.fuser_conv = nn.Sequential(*fusion_layers_conv)
        self.fuser = [self.fuser_att, self.fuser_mlp, self.fuser_conv]
        roi_heads = SeqRoIHeads(
            # OIM
            num_pids=cfg.MODEL.LOSS.LUT_SIZE,
            num_cq_size=cfg.MODEL.LOSS.CQ_SIZE,
            oim_momentum=cfg.MODEL.LOSS.OIM_MOMENTUM,
            oim_scalar=cfg.MODEL.LOSS.OIM_SCALAR,
            # GLCNet
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=cfg.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
            cxt_scene_extractor=self.cxt_scene_extractor if config.cxt_scene_enabled else None,
            cxt_group_extractor=self.cxt_group_extractor if config.cxt_group_enabled else None,
            fuser=self.fuser if config.use_fusion else [[], [], []],
            mps=self.mps if config.multi_part_matching else None
        )

        transform = GeneralizedRCNNTransform(
            min_size=cfg.INPUT.MIN_SIZE,
            max_size=cfg.INPUT.MAX_SIZE,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform

        # loss weights
        self.lw_rpn_reg = cfg.SOLVER.LW_RPN_REG
        self.lw_rpn_cls = cfg.SOLVER.LW_RPN_CLS
        self.lw_proposal_reg = cfg.SOLVER.LW_PROPOSAL_REG
        self.lw_proposal_cls = cfg.SOLVER.LW_PROPOSAL_CLS
        self.lw_box_reg = cfg.SOLVER.LW_BOX_REG
        self.lw_box_cls = cfg.SOLVER.LW_BOX_CLS
        self.lw_box_reid = cfg.SOLVER.LW_BOX_REID

    def inference(self, images, targets=None, query_img_as_gallery=False):
        """
        query_img_as_gallery: Set to True to detect all people in the query image.
            Meanwhile, the gt box should be the first of the detected boxes.
            This option serves CBGM.
        """
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        features = OrderedDict([["feat_res3", features["feat_res3"]]])

        if query_img_as_gallery:
            assert targets is not None
        
        proposals, _ = self.rpn(images, features, targets)
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
                if config.cxt_scene_enabled:
                    cxt_scene_proposalNum = torch.mean(self.roi_heads.x_embedding_head['cxt_scene'], dim=0).unsqueeze(0)
                    if config.nae_multi:
                        box_features['cxt_scene'] = cxt_scene_proposalNum
                    else:
                        box_features["feat_res4"] = torch.cat([box_features["feat_res4"], cxt_scene_proposalNum], 1)
                if config.cxt_group_enabled:
                    cxt_group_proposalNum = torch.mean(self.roi_heads.x_embedding_head['cxt_group'], dim=0).unsqueeze(0)
                    if config.nae_multi:
                        box_features['cxt_group'] = cxt_group_proposalNum
                    else:
                        box_features["feat_res4"] = torch.cat([box_features["feat_res4"], cxt_group_proposalNum], 1)
                if not config.nae_multi:
                    if self.fuser_att:
                        box_features["feat_res4"] = self.fuser_att(box_features["feat_res4"])
                    if self.fuser_mlp:
                        box_features["feat_res4"] = self.fuser_mlp(box_features["feat_res4"].squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
                    elif self.fuser_conv:
                        box_features["feat_res4"] = self.fuser_conv(box_features["feat_res4"].squeeze(-1)).unsqueeze(-1)
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
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        _, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # rename rpn losses to be consistent with detection losses
        proposal_losses["loss_rpn_reg"] = proposal_losses.pop("loss_rpn_box_reg")
        proposal_losses["loss_rpn_cls"] = proposal_losses.pop("loss_objectness")

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # apply loss weights
        losses["loss_rpn_reg"] *= self.lw_rpn_reg
        losses["loss_rpn_cls"] *= self.lw_rpn_cls
        losses["loss_proposal_reg"] *= self.lw_proposal_reg
        losses["loss_proposal_cls"] *= self.lw_proposal_cls
        losses["loss_box_reg"] *= self.lw_box_reg
        losses["loss_box_cls"] *= self.lw_box_cls
        losses["loss_box_reid"] *= self.lw_box_reid
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
                in_channels=[1024, config.nae_norm2_len]
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
            mps_channels = config.mps_channels if config.mps_channels else 2048
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
            in_channels=in_channels
        )
        self.reid_loss = OIMLoss(sum(config.nae_dims), num_pids, num_cq_size, oim_momentum, oim_scalar)
        self.faster_rcnn_predictor = faster_rcnn_predictor
        self.reid_head = reid_head
        # rename the method inherited from parent class
        self.postprocess_proposals = self.postprocess_detections
        self.cxt_scene_extractor = cxt_scene_extractor
        self.cxt_group_extractor = cxt_group_extractor
        self.fuser_att, self.fuser_mlp, self.fuser_conv = fuser

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
        if config.cxt_group_enabled:
            whether_box_has_person_lst_per_image = []
            for box_pid_label in box_pid_labels:
                if config.cxt_group_labelledOnly:
                    whether_box_has_person_lst = 0 < box_pid_label < 5555
                else:
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
                mps_feat_dict = self.mps(box_features)
            box_features = self.reid_head(box_features)
            if config.multi_part_matching:
                box_features.update(mps_feat_dict)
            box_regs = self.box_predictor(box_features["feat_res4"])
            if config.cxt:
                feat_res4_ori = box_features['feat_res4']
                if config.cxt_scene_enabled:
                    cxt_scene = self.cxt_scene_extractor(features['feat_res3']).squeeze(-1)
                    cxt_scene_proposalNum = torch.cat([
                        cxt_scene[idx_bs].unsqueeze(0).repeat(box.shape[0], 1, 1) for idx_bs, box in enumerate(boxes)
                    ], dim=0)   # I repeat the scene feature for box times to add the scene context to each box -- Equip all objects in this image with the scene context here.
                    if config.nae_multi:
                        box_features['cxt_scene'] = cxt_scene_proposalNum.unsqueeze(-1)
                    else:
                        box_features["feat_res4"] = torch.cat([box_features["feat_res4"], cxt_scene_proposalNum.unsqueeze(-1)], 1)
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
                        # cxt_group_per_image.append(torch.sum(feat_res4 * whether_box_has_person_lst, dim=0).unsqueeze(0) / (torch.sum(whether_box_has_person_lst) + 1e-5))
                        feat_persons = feat_res4[whether_box_has_person_lst]
                        feat_persons_squeezed = torch.mean(feat_persons, dim=0)
                        feat_persons_squeezed = feat_persons_squeezed.unsqueeze(0)
                        cxt_group_per_image.append(feat_persons_squeezed)
                    cxt_group_per_image = torch.cat(cxt_group_per_image, dim=0)
                    cxt_group_per_image = self.cxt_group_extractor(cxt_group_per_image)
                    cxt_group_proposalNum = torch.cat([
                        cxt_group_per_image[idx_bs].repeat(box.shape[0], 1, 1, 1) for idx_bs, box in enumerate(boxes)
                    ], dim=0)
                    if config.nae_multi:
                        box_features['cxt_group'] = cxt_group_proposalNum
                    else:
                        box_features["feat_res4"] = torch.cat([box_features["feat_res4"], cxt_group_proposalNum], 1)
                if not config.nae_multi:
                    if self.fuser_att:
                        box_features["feat_res4"] = self.fuser_att(box_features["feat_res4"])
                    if self.fuser_mlp:
                        box_features["feat_res4"] = self.fuser_mlp(box_features["feat_res4"].squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
                    elif self.fuser_conv:
                        box_features["feat_res4"] = self.fuser_conv(box_features["feat_res4"].squeeze(-1)).unsqueeze(-1)
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
                    # Use the contexts to enhance the features of surrounding persons.
                    if config.cxt_scene_enabled:
                        cxt_scene_proposalNum = torch.mean(self.x_embedding_head['cxt_scene'], dim=0).unsqueeze(0)
                        if config.nae_multi:
                            gt_box_features['cxt_scene'] = cxt_scene_proposalNum
                        else:
                            gt_box_features["feat_res4"] = torch.cat([gt_box_features["feat_res4"], cxt_scene_proposalNum], 1)
                    if config.cxt_group_enabled:
                        cxt_group_proposalNum = torch.mean(self.x_embedding_head['cxt_group'], dim=0).unsqueeze(0)
                        if config.nae_multi:
                            gt_box_features['cxt_group'] = cxt_group_proposalNum
                        else:
                            gt_box_features["feat_res4"] = torch.cat([gt_box_features["feat_res4"], cxt_group_proposalNum], 1)
                if not config.nae_multi:
                    if self.fuser_att:
                        gt_box_features["feat_res4"] = self.fuser_att(gt_box_features["feat_res4"])
                    if self.fuser_mlp:
                        gt_box_features["feat_res4"] = self.fuser_mlp(gt_box_features["feat_res4"].squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
                    elif self.fuser_conv:
                        gt_box_features["feat_res4"] = self.fuser_conv(gt_box_features["feat_res4"].squeeze(-1)).unsqueeze(-1)
                embeddings, _ = self.embedding_head(gt_box_features if config.nae_mix_res3 or config.nae_multi else OrderedDict([['feat_res4', gt_box_features["feat_res4"]]]))
                gt_det = {"boxes": targets[0]["boxes"], "embeddings": embeddings}

            box_embeddings, box_cls_scores = self.embedding_head(x_embedding_head)
            if box_cls_scores.dim() == 0:
                box_cls_scores = box_cls_scores.unsqueeze(0)
        else:
            if not self.training and query_img_as_gallery:
                cws = False
                gt_boxes = [targets[0]["boxes"]]
                gt_box_features = self.box_roi_pool(features, gt_boxes, image_shapes)
                if config.multi_part_matching:
                    mps_feat_dict = self.mps(gt_box_features)
                gt_box_features = self.reid_head(gt_box_features)
                if config.multi_part_matching:
                    gt_box_features.update(mps_feat_dict)
                if not config.nae_multi:
                    if self.fuser_att:
                        gt_box_features["feat_res4"] = self.fuser_att(gt_box_features["feat_res4"])
                    if self.fuser_mlp:
                        gt_box_features["feat_res4"] = self.fuser_mlp(gt_box_features["feat_res4"].squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
                    elif self.fuser_conv:
                        gt_box_features["feat_res4"] = self.fuser_conv(gt_box_features["feat_res4"].squeeze(-1)).unsqueeze(-1)
                embeddings, _ = self.embedding_head(gt_box_features if config.nae_mix_res3 or config.nae_multi else OrderedDict([['feat_res4', gt_box_features["feat_res4"]]]))
                gt_det = {"boxes": targets[0]["boxes"], "embeddings": embeddings}
            # no detection predicted by Faster R-CNN head in test phase
            assert not self.training
            boxes = gt_det["boxes"] if gt_det else torch.zeros(0, 4)
            labels = torch.ones(1).type_as(boxes) if gt_det else torch.zeros(0)
            scores = torch.ones(1).type_as(boxes) if gt_det else torch.zeros(0)
            embeddings = gt_det["embeddings"] if gt_det else torch.zeros(0, sum(config.nae_dims))
            return [dict(boxes=boxes, labels=labels, scores=scores, embeddings=embeddings)], []

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
            if torch.isnan(loss_box_reid):
                print('Met nan in <loss_box_reid> during training.')
                print(box_embeddings.shape)
                print(len(box_pid_labels))
                print(loss_box_reid)
                loss_box_reid = 0.0
            losses.update(loss_box_reid=loss_box_reid)
            # the return value of cross_entropy() in pytorch was changed from `0.0 to nan` when target.numel() == 0.
            # reference: https://github.com/pytorch/pytorch/issues/50224
        else:
            # The IoUs of these boxes are higher than that of proposals,
            # so a higher NMS threshold is needed
            orig_thresh = self.nms_thresh
            self.nms_thresh = 0.5
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
            # Fist Classification Score (FCS)
            pred_scores = fcs[0]
        else:
            pred_scores = torch.sigmoid(class_logits)
        if cws:
            # Confidence Weighted Similarity (CWS)
            embeddings = embeddings * pred_scores.view(-1, 1)

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
            embeddings = embeddings.reshape(-1, self.embedding_head.dim if isinstance(self.embedding_head.dim, int) else sum(self.embedding_head.dim))

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = (
                boxes[inds],
                scores[inds],
                labels[inds],
                embeddings[inds],
            )

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            if gt_det is not None:
                # include GT into the detection results
                boxes = torch.cat((boxes, gt_det["boxes"]), dim=0)
                labels = torch.cat((labels, torch.tensor([1.0]).to(device)), dim=0)
                scores = torch.cat((scores, torch.tensor([1.0]).to(device)), dim=0)
                embeddings = torch.cat((embeddings, gt_det["embeddings"]), dim=0)

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
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

    def __init__(self, featmap_names=["feat_res3", "feat_res4"], in_channels=[1024, 2048], dim=config.nae_dims):
        super(NormAwareEmbedding, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = dim

        self.projectors = nn.ModuleDict()
        if len(dim) == 1:
            indv_dims = self._split_embedding_dim(self.dim[0])
        else:
            indv_dims = dim
            if config.multi_part_matching:
                num_parts = 1 + 2 + 3       # 1 + 2 + 3 part for multi_part_matching, refer to the definition of MultiPartSpliter class.
                for _ in range(num_parts):
                    indv_dims.append(config.mps_norm_len)

        for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            proj = nn.Sequential(nn.Linear(in_channel, indv_dim), nn.BatchNorm1d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        self.rescaler = []
        if config.bn_feature_seperately:
            for _ in dim:
                self.rescaler.append(nn.BatchNorm1d(1, affine=True))
        else:
            self.rescaler = nn.BatchNorm1d(1, affine=True)

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
        if len(featmaps) == 1:
            for k, v in featmaps.items():
                pass
            v = self._flatten_fc_input(v)
            embeddings = self.projectors[k](v)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms
        else:
            if config.bn_feature_seperately:
                embeddings = []
                norms = []
                for idx_fm, (k, v) in enumerate(featmaps.items()):
                    v = self._flatten_fc_input(v)
                    proj_feat = self.projectors[k](v)
                    norm = proj_feat.norm(2, 1, keepdim=True)
                    embedding = proj_feat / norm.expand_as(proj_feat).clamp(min=1e-12)
                    norm = self.rescaler[idx_fm](norm).squeeze()
                    embeddings.append(embedding)
                    norms.append(norm)
                embeddings = torch.cat(embeddings, dim=1)
                norms = torch.cat(norms, dim=1)
            else:
                outputs = []
                for k, v in featmaps.items():
                    v = self._flatten_fc_input(v)
                    proj_feat = self.projectors[k](v)
                    outputs.append(proj_feat)
                embeddings = torch.cat(outputs, dim=1)
                norms = embeddings.norm(2, 1, keepdim=True)
                embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
                norms = self.rescaler(norms).squeeze()
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
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes), nn.BatchNorm1d(4 * num_classes)
            )
            init.normal_(self.bbox_pred[0].weight, std=0.01)
            init.normal_(self.bbox_pred[1].weight, std=0.01)
            init.constant_(self.bbox_pred[0].bias, 0)
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


def detection_losses(
    proposal_cls_scores,
    proposal_regs,
    proposal_labels,
    proposal_reg_targets,
    box_cls_scores,
    box_regs,
    box_labels,
    box_reg_targets,
):
    proposal_labels = torch.cat(proposal_labels, dim=0)
    box_labels = torch.cat(box_labels, dim=0)
    proposal_reg_targets = torch.cat(proposal_reg_targets, dim=0)
    box_reg_targets = torch.cat(box_reg_targets, dim=0)

    loss_proposal_cls = F.cross_entropy(proposal_cls_scores, proposal_labels)
    loss_box_cls = F.binary_cross_entropy_with_logits(box_cls_scores, box_labels.float())

    # get indices that correspond to the regression targets for the
    # corresponding ground truth labels, to be used with advanced indexing
    sampled_pos_inds_subset = torch.nonzero(proposal_labels > 0).squeeze(1)
    labels_pos = proposal_labels[sampled_pos_inds_subset]
    N = proposal_cls_scores.size(0)
    proposal_regs = proposal_regs.reshape(N, -1, 4)

    loss_proposal_reg = F.smooth_l1_loss(
        proposal_regs[sampled_pos_inds_subset, labels_pos],
        proposal_reg_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_proposal_reg = loss_proposal_reg / proposal_labels.numel()

    sampled_pos_inds_subset = torch.nonzero(box_labels > 0).squeeze(1)
    labels_pos = box_labels[sampled_pos_inds_subset]
    N = box_cls_scores.size(0)
    box_regs = box_regs.reshape(N, -1, 4)

    loss_box_reg = F.smooth_l1_loss(
        box_regs[sampled_pos_inds_subset, labels_pos],
        box_reg_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_box_reg = loss_box_reg / box_labels.numel()

    return dict(
        loss_proposal_cls=loss_proposal_cls,
        loss_proposal_reg=loss_proposal_reg,
        loss_box_cls=loss_box_cls,
        loss_box_reg=loss_box_reg,
    )

