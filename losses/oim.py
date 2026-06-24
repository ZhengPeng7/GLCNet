import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import autograd, nn
from configs import config


class OIM(autograd.Function):
    """Projects features against the [lut; cq] memory-bank snapshot. The bank UPDATE lives in
    OIMLoss._update_banks (not here); this Function only carries the autocast-disabled fp32 matmul
    and its gradient. `lut_cq` is a cat() snapshot, so the post-loss in-place bank update never
    corrupts the operand saved for grad_inputs (keeps the gradient identical to the pre-update bank)."""
    @staticmethod
    def forward(ctx, inputs, lut_cq):
        ctx.save_for_backward(inputs, lut_cq)
        # Disable autocast so the mm() below truly runs in float32. Under an active autocast context
        # the .float() casts are silently defeated: autocast re-intercepts mm at the dispatch level
        # and recasts operands back to bf16. cross_entropy needs fp32 logits.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return inputs.float().mm(lut_cq.float().t())

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, lut_cq = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            # Keep gradients in float32 for numerical stability - DO NOT convert back to bf16
            grad_inputs = grad_outputs.float().mm(lut_cq.float())
        return grad_inputs, None


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
        # header_cq: circular-queue write pointer. Stays consistent across DDP ranks because every
        # rank applies the IDENTICAL all-gathered bank update in _update_banks (not via broadcast_buffers).
        self.register_buffer("header_cq", torch.zeros((), dtype=torch.long))
        self.ignore_index = 5554

    def _update_banks(self, inputs, label):
        """Update lut (labeled momentum) and cq (unlabeled circular queue) from this step's samples.
        Under DDP every rank all-gathers all ranks' (features, labels) and applies the SAME global
        update, so all bank copies stay byte-identical across ranks (broadcast_buffers then unneeded);
        single-GPU is a pure passthrough. Runs under no_grad AFTER the projection snapshot, so it never
        affects grad_inputs (which is computed from the pre-update [lut; cq] snapshot)."""
        with torch.no_grad():
            x_all = inputs.detach()
            y_all = label.detach()
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                ws = dist.get_world_size()
                # variable per-rank sample counts -> gather sizes, pad to max, all_gather, slice back
                n = torch.tensor([x_all.size(0)], device=x_all.device)
                ns = [torch.zeros_like(n) for _ in range(ws)]
                dist.all_gather(ns, n)
                sizes = [int(s.item()) for s in ns]
                m = max(max(sizes), 1)
                xp = x_all.new_zeros(m, self.num_features)
                xp[: x_all.size(0)] = x_all
                yp = y_all.new_full((m,), -1)
                yp[: y_all.size(0)] = y_all
                xs = [torch.zeros_like(xp) for _ in range(ws)]
                ys = [torch.zeros_like(yp) for _ in range(ws)]
                dist.all_gather(xs, xp)
                dist.all_gather(ys, yp)
                # canonical rank order; strip padding via per-rank size
                x_all = torch.cat([xs[r][: sizes[r]] for r in range(ws)], dim=0)
                y_all = torch.cat([ys[r][: sizes[r]] for r in range(ws)], dim=0)
            for x, y in zip(x_all, y_all):
                if y < self.num_pids:
                    # Compute in float32 for numerical stability with bf16/fp16
                    v = self.momentum * self.lut[y].float() + (1.0 - self.momentum) * x.float()
                    # Use larger clamp value (1e-4) for bf16 stability
                    self.lut[y] = (v / v.norm().clamp(min=1e-4)).to(self.lut.dtype)
                else:
                    h = int(self.header_cq)
                    self.cq[h] = x.to(self.cq.dtype)
                    self.header_cq.fill_((h + 1) % self.num_unlabeled)

    def forward(self, inputs, roi_label):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1

        inds = label >= 0
        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)

        # snapshot the memory bank (cat -> new tensor); the post-loss in-place update below leaves it untouched
        lut_cq = torch.cat([self.lut, self.cq], dim=0)
        projected = OIM.apply(inputs, lut_cq)
        projected *= self.oim_scalar

        # the return value of cross_entropy() in pytorch was changed from `0.0 to nan` when target.numel() == 0.
        # reference: https://github.com/pytorch/pytorch/issues/50224
        # label.numel() == 0 (all-background image) would also crash label.min()/max(); guard it.
        if label.numel() == 0 or (label.min() == label.max() and label.max() == self.ignore_index):
            loss_oim = projected.sum() * 0.0
        else:
            # Cast to float32 for numerical stability with bf16/fp16
            loss_oim = F.cross_entropy(projected.float(), label, ignore_index=self.ignore_index)

        # update the memory banks (DDP-synced via all-gather) AFTER the projection/loss
        self._update_banks(inputs, label)
        return loss_oim
