import torch
import torch.nn.functional as F


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

    # Cast to float32 for numerical stability with bf16/fp16
    loss_proposal_cls = F.cross_entropy(proposal_cls_scores.float(), proposal_labels)
    loss_box_cls = F.binary_cross_entropy_with_logits(box_cls_scores.float(), box_labels.float())

    # get indices that correspond to the regression targets for the
    # corresponding ground truth labels, to be used with advanced indexing
    sampled_pos_inds_subset = torch.nonzero(proposal_labels > 0).squeeze(1)
    labels_pos = proposal_labels[sampled_pos_inds_subset]
    N = proposal_cls_scores.size(0)
    proposal_regs = proposal_regs.reshape(N, -1, 4)

    # Cast to float32 for numerical stability with bf16/fp16
    loss_proposal_reg = F.smooth_l1_loss(
        proposal_regs[sampled_pos_inds_subset, labels_pos].float(),
        proposal_reg_targets[sampled_pos_inds_subset].float(),
        reduction="sum",
    )
    loss_proposal_reg = loss_proposal_reg / proposal_labels.numel()

    sampled_pos_inds_subset = torch.nonzero(box_labels > 0).squeeze(1)
    labels_pos = box_labels[sampled_pos_inds_subset]
    N = box_cls_scores.size(0)
    box_regs = box_regs.reshape(N, -1, 4)

    # Cast to float32 for numerical stability with bf16/fp16
    loss_box_reg = F.smooth_l1_loss(
        box_regs[sampled_pos_inds_subset, labels_pos].float(),
        box_reg_targets[sampled_pos_inds_subset].float(),
        reduction="sum",
    )
    loss_box_reg = loss_box_reg / box_labels.numel()

    return dict(
        loss_proposal_cls=loss_proposal_cls,
        loss_proposal_reg=loss_proposal_reg,
        loss_box_cls=loss_box_cls,
        loss_box_reg=loss_box_reg,
    )
