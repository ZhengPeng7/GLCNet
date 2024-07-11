import os
import argparse
import datetime
import time

import torch
import torch.utils.data

from datasets import build_test_loader, build_train_loader
from defaults import get_default_cfg
from engine import evaluate_performance, train_one_epoch
from models.glcnet import GLCNet
from utils.utils import mkdir, resume_from_ckpt, save_on_master, set_random_seed
from config import Config


config = Config()


def main(args):
    cfg = get_default_cfg()
    cfg.set_new_allowed(True)
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    if cfg.INPUT.DATASET == "CUHK-SYSU":
        cfg.INPUT.DATA_ROOT = os.path.join(cfg.INPUT.DATA_ROOT_PS, 'cuhk_sysu')
    elif cfg.INPUT.DATASET == "PRW":
        cfg.INPUT.DATA_ROOT = os.path.join(cfg.INPUT.DATA_ROOT_PS, 'prw')
    elif cfg.INPUT.DATASET == "MVN":
        cfg.INPUT.DATA_ROOT = os.path.join(cfg.INPUT.DATA_ROOT_PS, 'MovieNet-PS')
    cfg.freeze()

    device = torch.device(cfg.DEVICE)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    print("Creating model")
    model = GLCNet(cfg)
    model.to(device)

    print("Loading data")
    gallery_loader, query_loader = build_test_loader(cfg)

    if args.eval:
        assert args.ckpt, "--ckpt must be specified when --eval enabled"
        resume_from_ckpt(args.ckpt, model)
        evaluate_performance(
            model,
            gallery_loader,
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
        )
        exit(0)

    train_loader = build_train_loader(cfg)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config.lr * (cfg.INPUT.BATCH_SIZE_TRAIN / 3),    # adapt the lr linearly,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1
    )

    start_epoch = 1
    if args.ckpt and not args.eval:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        print('Resuming from', args.ckpt)
        # Resume from models pre-trained on MovieNet-PS. Otherwise, assign the return value to `start_epoch`.
        _ = resume_from_ckpt(args.ckpt, model, optimizer, lr_scheduler) + 1

    print("Creating output folder")
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = os.path.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config is saved to {path}")
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = os.path.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        print(f"TensorBoard files are saved to {tf_log_path}")

    print("Start training")
    start_time = time.time()
    mAP_top1_lst = [0]
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS+1):
        print('Epoch {}:'.format(epoch))
        train_one_epoch(cfg, model, optimizer, train_loader, device, epoch, lr_scheduler, tfboard)
        lr_scheduler.step()

        if(
            epoch >= cfg.SOLVER.MAX_EPOCHS or
            (
                epoch % cfg.EVAL_PERIOD == 0 and
                epoch > max(cfg.SOLVER.LR_DECAY_MILESTONES[-1], cfg.SOLVER.MAX_EPOCHS-5)
            ) or
            (
                'MVN' in cfg.INPUT.DATASET and
                (epoch % 5 == 0 or epoch > max(cfg.SOLVER.LR_DECAY_MILESTONES[-1], cfg.SOLVER.MAX_EPOCHS-10))
            )
        ):
            mAP, top1 = evaluate_performance(
                model,
                gallery_loader,
                query_loader,
                device,
                use_gt=cfg.EVAL_USE_GT,
                use_cache=cfg.EVAL_USE_CACHE,
                use_cbgm=cfg.EVAL_USE_CBGM,
            )
        else:
            mAP, top1 = 0, 0
        mAP_top1 = mAP + top1 * 0.5     # mAP is more important
        mAP_top1_lst.append(mAP_top1)
        if mAP_top1 > max(mAP_top1_lst[:-1]):
            print('Saving the best model with mAP={:.3f}, top-1={:.3f} ...'.format(mAP, top1))
            save_on_master(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                },
                [os.path.join(output_dir, f"epoch_{epoch}.pth"), os.path.join(output_dir, "epoch_best.pth")][1],
            )

    if tfboard:
        tfboard.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file.")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the performance of a given checkpoint."
    )
    parser.add_argument("--ckpt", default='', help="Path to checkpoint to resume or evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    args = parser.parse_args()
    main(args)
