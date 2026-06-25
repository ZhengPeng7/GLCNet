import os
import argparse
import datetime
import time

import torch
import torch.utils.data

pytorch_version_tuple = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:3])
if pytorch_version_tuple >= (2, 5, 0):
    alloc_conf_key = 'PYTORCH_ALLOC_CONF' if pytorch_version_tuple >= (2, 9, 0) else 'PYTORCH_CUDA_ALLOC_CONF'
    os.environ[alloc_conf_key] = 'expandable_segments:True'

from datasets import build_test_loader, build_train_loader
from models.glcnet import GLCNet
from utils.utils import mkdir, set_random_seed, MetricLogger, SmoothedValue
from configs import config
from eval import evaluate_performance


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import accelerate
accelerator = accelerate.Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=1,
    kwargs_handlers=[
        accelerate.utils.InitProcessGroupKwargs(backend="nccl", timeout=datetime.timedelta(seconds=3600)),
        # broadcast_buffers=False: OIM memory banks (lut/cq/header_cq) are kept in sync across ranks
        # manually via all-gather in OIMLoss._update_banks; the default rank0 re-broadcast would
        # instead DISCARD ranks 1..N's bank updates. All other buffers are identical across ranks.
        accelerate.utils.DistributedDataParallelKwargs(find_unused_parameters=False, broadcast_buffers=False),
        accelerate.utils.GradScalerKwargs(backoff_factor=0.5)],
)


def build_optimizer_param_groups(model):
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if config.dino_v3_stage_unfreeze_active:
        def is_dino_backbone(name):
            return name.replace('_orig_mod.', '').startswith('backbone.bb.')

        dino_params = [param for name, param in trainable if is_dino_backbone(name)]
        other_params = [param for name, param in trainable if not is_dino_backbone(name)]
        if not dino_params:
            raise RuntimeError('DINOv3 stage unfreeze is active, but no trainable backbone parameters were found.')
        return [
            {'params': other_params, 'lr': config.lr},
            {'params': dino_params, 'lr': config.lr * config.dino_v3_active_lr_mult},
        ]
    return [param for _, param in trainable]


def train_one_epoch(model, optimizer, data_loader, epoch, lr_scheduler, output_dir, tb_writer=None, iter_based_scheduler=False):
    """
    Args:
        iter_based_scheduler: If True, lr_scheduler.step() is called per iteration.
                              If False, warmup is handled internally and lr_scheduler is stepped per epoch.
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ", log_file=os.path.join(output_dir, 'training.log'), append=(epoch > 1), main_process=accelerator.is_main_process)
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    # For MultiStepLR: warmup is handled internally (per-iteration in first epoch)
    warmup_scheduler = None
    if not iter_based_scheduler and config.warmup_enabled and epoch <= config.warmup_epochs:
        warmup_iters = len(data_loader)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters
        )

    for i, (images, targets) in enumerate(
        metric_logger.log_every(data_loader, config.disp_period, header)
    ):
        with accelerator.accumulate(model):
            with accelerator.autocast():
                loss_dict = model(images, targets)
                # Sum losses in float32 for numerical stability with bf16/fp16
                loss = sum(loss_v.float() for loss_v in loss_dict.values())
                loss_value = loss.item()
            accelerator.backward(loss)
            clip_val = config.clip_gradients if config.clip_gradients > 0 else 1.0
            accelerator.clip_grad_value_(model.parameters(), clip_val)
            optimizer.step()
            optimizer.zero_grad()

        # Update lr scheduler
        if iter_based_scheduler:
            lr_scheduler.step()
        elif warmup_scheduler is not None:
            warmup_scheduler.step()

        # Convert loss values to float for logging precision with bf16/fp16
        loss_dict_float = {k: v.float().item() if hasattr(v, 'float') else v for k, v in loss_dict.items()}
        metric_logger.update(loss=loss_value, **loss_dict_float)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        if tb_writer and accelerator.is_main_process:
            step = (epoch - 1) * len(data_loader) + i
            tb_writer.add_scalar('Hyper-Params/lr', optimizer.param_groups[0]['lr'], step)
            for k, v in loss_dict_float.items():
                tb_writer.add_scalar(f'Loss/{k}', v, step)


def main(args):
    # Override output_dir if --ckpt_dir is specified
    output_dir = args.ckpt_dir if args.ckpt_dir else config.output_dir
    epochs = config.max_epochs

    device = torch.device(config.device)
    if config.seed >= 0:
        set_random_seed(config.seed)

    print("Creating model")
    model = GLCNet()
    model.to(device)

    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')

    print("Loading data")
    train_loader = build_train_loader(config)
    gallery_loader, query_loader = build_test_loader(config)

    # Handle --cbgm flag
    use_cbgm = args.cbgm or config.eval_use_cbgm
    print('use_cbgm:', use_cbgm)

    params = build_optimizer_param_groups(model)
    if config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            params,
            lr=config.lr,
            momentum=config.sgd_momentum,
            weight_decay=config.sgd_weight_decay,
            nesterov=config.sgd_nesterov,
        )
    elif config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            params,
            lr=config.lr,
            weight_decay=config.adamw_weight_decay,
        )
    elif config.optimizer == 'Lion':
        from lion_pytorch import Lion
        optimizer = Lion(
            params,
            lr=config.lr,
            weight_decay=config.lion_weight_decay,
            use_triton=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # accelerate preparation
    # Only train_loader is sharded across processes. gallery/query loaders stay full so the
    # main-process eval (below) sees the complete dataset on every process count (1 or N).
    train_loader, model, optimizer = accelerator.prepare(train_loader, model, optimizer)

    # Build LR scheduler
    iters_per_epoch = len(train_loader)
    iter_based_scheduler = (config.scheduler == 'CosineAnnealingLR')

    if config.scheduler == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.multistep_milestones, gamma=config.multistep_gamma
        )
    elif config.scheduler == 'CosineAnnealingLR':
        warmup_iters = config.warmup_epochs * iters_per_epoch if config.warmup_enabled else 0
        cosine_iters = (config.max_epochs - config.warmup_epochs) * iters_per_epoch if config.warmup_enabled else config.max_epochs * iters_per_epoch

        if config.warmup_enabled:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_iters, eta_min=config.cosine_eta_min
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iters]
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_iters, eta_min=config.cosine_eta_min
            )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")

    print("Creating output folder")
    mkdir(output_dir)
    print(f"Output directory: {output_dir}")
    tb_writer = None
    if config.tf_board and accelerator.is_main_process:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = os.path.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tb_writer = SummaryWriter(log_dir=tf_log_path)
        print(f"TensorBoard files are saved to {tf_log_path}")

    print("Start training")
    start_time = time.time()
    main_logger = MetricLogger(log_file=os.path.join(output_dir, 'training.log'), append=True, main_process=accelerator.is_main_process)
    best_score, best_epoch, best_mAP, best_top1 = 0.0, 0, 0.0, 0.0
    for epoch in range(1, epochs+1):
        main_logger.log('Epoch {}:'.format(epoch))
        train_one_epoch(model, optimizer, train_loader, epoch, lr_scheduler, output_dir, tb_writer, iter_based_scheduler)
        # For MultiStepLR (epoch-based): step after each epoch, skip during warmup
        if not iter_based_scheduler:
            if not config.warmup_enabled or epoch > config.warmup_epochs:
                lr_scheduler.step()

        if(
            epoch >= epochs or
            (
                epoch % config.eval_period == 0 and
                epoch >= min(config.multistep_milestones[0], epochs-3)
            ) or
            (
                'MVN' in config.task and
                (epoch % 5 == 0 or epoch >= min(config.multistep_milestones[0], epochs-10))
            )
        ):
            # Eval (inference + metrics) runs on the main process only, over the full (un-sharded)
            # gallery/query loaders, using the unwrapped model so no DDP collectives fire during the
            # solo eval. Other ranks skip straight to the unconditional epoch barrier below.
            if accelerator.is_main_process:
                mAP, top1 = evaluate_performance(
                    accelerator.unwrap_model(model),
                    gallery_loader,
                    query_loader,
                    device,
                    use_gt=config.eval_use_gt,
                    use_cache=False,
                    use_cbgm=use_cbgm,
                )
                score = round(float(mAP + top1 * 0.5), 4)
                if score > best_score:
                    best_score, best_epoch, best_mAP, best_top1 = score, epoch, mAP, top1
                    main_logger.log(f'New best! Epoch {epoch}: mAP={mAP:.4f}, top-1={top1:.4f}, score={score:.4f}')
                    torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(output_dir, "epoch_best.pth"))
                else:
                    main_logger.log(f'Epoch {epoch}: mAP={mAP:.4f}, top-1={top1:.4f}, score={score:.4f} (best: Epoch {best_epoch}, score={best_score:.4f})')
        if accelerator.is_main_process:
            torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(output_dir, f"epoch_last.pth"))
        accelerator.wait_for_everyone()

    if tb_writer:
        tb_writer.close()
    accelerator.end_training()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    main_logger.log(f"Total training time {total_time_str}")
    if best_epoch > 0:
        main_logger.log(f"Best model: Epoch {best_epoch}, mAP={best_mAP:.4f}, top-1={best_top1:.4f}, score={best_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--ckpt_dir", default='', help="Path to save checkpoints.")
    parser.add_argument("--cbgm", action="store_true", help="Use CBGM algorithm for evaluation.")
    args = parser.parse_args()
    main(args)
