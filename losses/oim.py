import torch
import torch.nn.functional as F
from torch import autograd, nn
from configs import config

class OIM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut, cq, header, momentum):
        ctx.save_for_backward(inputs, targets, lut, cq, header, momentum)
        # Keep outputs in float32 for numerical stability - DO NOT convert back to bf16
        # This is critical because cross_entropy needs high precision logits
        inputs_f = inputs.float()
        outputs_labeled = inputs_f.mm(lut.float().t())
        outputs_unlabeled = inputs_f.mm(cq.float().t())
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, lut, cq, header, momentum = ctx.saved_tensors

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            # Keep gradients in float32 for numerical stability - DO NOT convert back to bf16
            lut_cq = torch.cat([lut, cq], dim=0).float()
            grad_inputs = grad_outputs.float().mm(lut_cq)

        for x, y in zip(inputs, targets):
            if y < len(lut):
                # Compute in float32 for numerical stability with bf16/fp16
                lut_y = lut[y].float()
                x_float = x.float()
                momentum_val = momentum.float()
                lut_y = momentum_val * lut_y + (1.0 - momentum_val) * x_float
                # Use larger clamp value (1e-4) for bf16 stability
                lut[y] = (lut_y / lut_y.norm().clamp(min=1e-4)).to(lut.dtype)
            else:
                cq[header] = x
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None


def oim(inputs, targets, lut, cq, header, momentum=0.5):
    return OIM.apply(inputs, targets, lut, cq, torch.tensor(header, dtype=torch.long), torch.tensor(momentum, dtype=torch.float32))


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        if config.mixed_precision == 'bf16':
            self.model_dtype = torch.bfloat16
        elif config.mixed_precision == 'fp16':
            self.model_dtype = torch.float16
        else:
            self.model_dtype = torch.float32

        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features, dtype=self.model_dtype))
        self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features, dtype=self.model_dtype))

        self.header_cq = 0
        self.ignore_index = 5554

    def forward(self, inputs, roi_label):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1

        inds = label >= 0
        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)

        projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)
        projected *= self.oim_scalar

        self.header_cq = (
            self.header_cq + (label >= self.num_pids).long().sum().item()
        ) % self.num_unlabeled
        # the return value of cross_entropy() in pytorch was changed from `0.0 to nan` when target.numel() == 0.
        # reference: https://github.com/pytorch/pytorch/issues/50224
        if label.min() == label.max() and label.max() == self.ignore_index:
            loss_oim = projected.sum() * 0.0
        else:
            # Cast to float32 for numerical stability with bf16/fp16
            loss_oim = F.cross_entropy(projected.float(), label, ignore_index=self.ignore_index)
        return loss_oim
