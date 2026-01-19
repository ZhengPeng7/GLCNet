import os
import argparse
import datetime
import time

import torch
import torch.utils.data

from datasets import build_test_loader, build_train_loader
from models.glcnet import GLCNet
from utils.utils import mkdir, load_weights, set_random_seed
from configs import config


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import accelerate
accelerator = accelerate.Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=1,
    kwargs_handlers=[
        accelerate.utils.InitProcessGroupKwargs(backend="nccl", timeout=datetime.timedelta(seconds=3600)),
        accelerate.utils.DistributedDataParallelKwargs(find_unused_parameters=False),
        accelerate.utils.GradScalerKwargs(backoff_factor=0.5)],
)

from copy import deepcopy
from tqdm import tqdm
from eval_func import eval_detection, eval_search_cuhk, eval_search_prw, eval_search_mvn
from utils.utils import MetricLogger, SmoothedValue, mkdir, warmup_lr_scheduler


def train_one_epoch(model, optimizer, data_loader, epoch, lr_scheduler, output_dir, tfboard=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ", log_file=os.path.join(output_dir, 'training.log'))
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    # warmup learning rate in the first epoch
    if epoch <= 1:
        warmup_factor = config.warmup_factor
        warmup_iters = len(data_loader) - 1
        warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for i, (images, targets) in enumerate(
        metric_logger.log_every(data_loader, config.disp_period, header)
    ):
        with accelerator.autocast():
            loss_dict = model(images, targets)
            loss = sum(loss_v for loss_v in loss_dict.values())
            loss_value = loss.item()
        accelerator.backward(loss)

        with accelerator.accumulate(model):
            clip_val = config.clip_gradients if config.clip_gradients > 0 else 1.0
            accelerator.clip_grad_value_(model.parameters(), clip_val)
            optimizer.step()
            optimizer.zero_grad()

        if epoch <= 1:
            warmup_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        if tfboard:
            iter = (epoch - 1) * len(data_loader) + i
            for k, v in loss_dict.items():
                tfboard.add_scalars("train", {k: v}, iter)


@torch.no_grad()
def evaluate_performance(
    model, gallery_loader, query_loader, device, use_gt=False, use_cache=False, use_cbgm=False
):
    """
    Args:
        use_gt (bool, optional): Whether to use GT as detection results to verify the upper
                                bound of person search performance. Defaults to False.
        use_cache (bool, optional): Whether to use the cached features. Defaults to False.
        use_cbgm (bool, optional): Whether to use Context Bipartite Graph Matching algorithm.
                                Defaults to False.
    """
    model.eval()
    if use_cache:
        eval_cache = torch.load("data/eval_cache/eval_cache.pth", weights_only=False)
        gallery_dets = eval_cache["gallery_dets"]
        gallery_feats = eval_cache["gallery_feats"]
        query_dets = eval_cache["query_dets"]
        query_feats = eval_cache["query_feats"]
        query_box_feats = eval_cache["query_box_feats"]
    else:
        gallery_dets, gallery_feats = [], []
        print(f'Extracting gallery at {time.asctime(time.gmtime())} ...')
        for images, targets in tqdm(gallery_loader, ncols=0):
            if not use_gt:
                with accelerator.autocast():
                    outputs = model(images)
            else:
                boxes = targets[0]["boxes"]
                n_boxes = boxes.size(0)
                with accelerator.autocast():
                    embeddings = model(images, targets)
                outputs = [
                    {
                        "boxes": boxes,
                        "embeddings": torch.cat(embeddings),
                        "labels": torch.ones(n_boxes).to(device),
                        "scores": torch.ones(n_boxes).to(device),
                    }
                ]

            for images, output in zip(images, outputs):
                expand_ratio = 0
                if expand_ratio:
                    for idx_box, (box, image) in enumerate(zip(output["boxes"], images)):
                        x1, y1, x2, y2 = box
                        hei_image, wid_image = image.shape[-2:]
                        wid, hei = (x2 - x1) / 2, (y2 - y1) / 2
                        output["boxes"][idx_box][0] = max(0, x1 - wid * expand_ratio)
                        output["boxes"][idx_box][1] = max(0, y1 - wid * expand_ratio)
                        output["boxes"][idx_box][2] = min(wid_image-1, x2 + hei * expand_ratio)
                        output["boxes"][idx_box][3] = min(hei_image-1, y2 + hei * expand_ratio)
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                gallery_dets.append(box_w_scores.cpu().numpy())
                gallery_feats.append(output["embeddings"].cpu().numpy())

        # regarding query image as gallery to detect all people
        # i.e. query person + surrounding people (context information)
        query_dets, query_feats = [], []
        if use_cbgm:
            print(f'Extracting query context at {time.asctime(time.gmtime())} ...')
            for images, targets in tqdm(query_loader, ncols=0):
                # targets will be modified in the model, so deepcopy it
                with accelerator.autocast():
                    outputs = model(images, deepcopy(targets), query_img_as_gallery=True)

                # consistency check
                gt_box = targets[0]["boxes"].squeeze()
                assert (
                    gt_box - outputs[0]["boxes"][0]
                ).sum() <= 0.001, "GT box must be the first one in the detected boxes of query image"

                for output in outputs:
                    box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                    query_dets.append(box_w_scores.cpu().numpy())
                    query_feats.append(output["embeddings"].cpu().numpy())

        # extract the features of query boxes
        query_box_feats = []
        print(f'Extracting query at {time.asctime(time.gmtime())} ...')
        for images, targets in tqdm(query_loader, ncols=0):
            with accelerator.autocast():
                embeddings = model(images, targets, query_img_as_gallery=False)
            assert len(embeddings) == 1, "batch size in test phase should be 1"
            query_box_feats.append(embeddings[0].cpu().numpy())
        print(f'Finish feature extraction on gallery+query at {time.asctime(time.gmtime())} .')

        mkdir("data/eval_cache")
        save_dict = {
            "gallery_dets": gallery_dets,
            "gallery_feats": gallery_feats,
            "query_dets": query_dets,
            "query_feats": query_feats,
            "query_box_feats": query_box_feats,
        }
        torch.save(save_dict, "data/eval_cache/eval_cache.pth")
    try:
        eval_detection(gallery_loader.dataset, gallery_dets, det_thresh=0.01)
        if gallery_loader.dataset.name == "CUHK-SYSU":
            ret = eval_search_cuhk(
                gallery_loader.dataset, query_loader.dataset, gallery_dets, gallery_feats, query_box_feats, query_dets, query_feats,
                cbgm=use_cbgm, gallery_size=100,
            )
        elif gallery_loader.dataset.name == "PRW":
            ret = eval_search_prw(
                gallery_loader.dataset, query_loader.dataset, gallery_dets, gallery_feats, query_box_feats, query_dets, query_feats,
                cbgm=use_cbgm,
            )
        elif gallery_loader.dataset.name == "MVN":
            ret = eval_search_mvn(
                gallery_loader.dataset, query_loader.dataset, gallery_dets, gallery_feats, query_box_feats, query_dets, query_feats,
                cbgm=use_cbgm, gallery_size=config.mvn_gallery_size,
            )
        mAP = ret["mAP"]
        top1 = ret["accs"][0]
    except Exception as e:
        print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        mAP = 0
        top1 = 0
    return mAP, top1


def main(args):
    # Override output_dir if --ckpt_dir is specified
    output_dir = args.ckpt_dir if args.ckpt_dir else config.output_dir
    epochs = config.max_epochs

    device = torch.device(config.device)
    if config.seed >= 0:
        set_random_seed(config.seed)

    print("Creating model")
    model = GLCNet(config)
    model.to(device)

    if config.compile:
        model = torch.compile(model, mode=['default', 'reduce-overhead', 'max-autotune'][0])
    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')

    print("Loading data")
    gallery_loader, query_loader = build_test_loader(config)

    # Handle --cbgm flag
    use_cbgm = args.cbgm or config.eval_use_cbgm

    if args.eval:
        assert args.ckpt, "--ckpt must be specified when --eval enabled"
        load_weights(args.ckpt, model)
        evaluate_performance(
            model,
            gallery_loader,
            query_loader,
            device,
            use_gt=config.eval_use_gt,
            use_cache=True,
            use_cbgm=use_cbgm,
        )
        exit(0)

    train_loader = build_train_loader(config)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config.lr_scaled,
        momentum=config.sgd_momentum,
        weight_decay=config.sgd_weight_decay,
    )

    # accelerate preparation
    train_loader, gallery_loader, query_loader, model, optimizer = accelerator.prepare(train_loader, gallery_loader, query_loader, model, optimizer)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.lr_decay_milestones, gamma=config.lr_decay_rate
    )

    start_epoch = 1
    if args.ckpt and not args.eval:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        print('Resuming from', args.ckpt)
        # Resume from models pre-trained on MovieNet-PS. Otherwise, assign the return value to `start_epoch`.
        _ = load_weights(args.ckpt, model) + 1

    print("Creating output folder")
    mkdir(output_dir)
    print(f"Output directory: {output_dir}")
    tfboard = None
    if config.tf_board:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = os.path.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        print(f"TensorBoard files are saved to {tf_log_path}")

    print("Start training")
    start_time = time.time()
    mAP_top1_lst = []
    for epoch in range(start_epoch, epochs+1):
        print('Epoch {}:'.format(epoch))
        train_one_epoch(model, optimizer, train_loader, epoch, lr_scheduler, output_dir, tfboard)
        lr_scheduler.step()

        if(
            epoch >= epochs or
            (
                epoch % config.eval_period == 0 and
                epoch >= min(config.lr_decay_milestones[0], epochs-3)
            ) or
            (
                'MVN' in config.task and
                (epoch % 5 == 0 or epoch >= min(config.lr_decay_milestones[0], epochs-10))
            )
        ):
            mAP, top1 = evaluate_performance(
                model,
                gallery_loader,
                query_loader,
                device,
                use_gt=config.eval_use_gt,
                use_cache=False,
                use_cbgm=use_cbgm,
            )
        else:
            mAP, top1 = 0, 0
        mAP_top1 = round(float(mAP + top1 * 0.5), 4)
        mAP_top1_lst.append(mAP_top1)
        print(f'mAP_top1_lst: {mAP_top1_lst}')
        if mAP_top1 > max(mAP_top1_lst[:-1] + [0]):
            print('Saving the best model with mAP={:.3f}, top-1={:.3f} ...'.format(round(mAP, 3), round(top1, 3)))
            torch.save(model.state_dict(), os.path.join(output_dir, "epoch_best.pth"))
        torch.save(model.state_dict(), os.path.join(output_dir, f"epoch_last.pth"))

    if tfboard:
        tfboard.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--ckpt_dir", default='', help="Path to save checkpoints.")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the performance of a given checkpoint."
    )
    parser.add_argument("--ckpt", default='', help="Path to checkpoint to resume or evaluate.")
    parser.add_argument("--cbgm", action="store_true", help="Use CBGM algorithm for evaluation.")
    args = parser.parse_args()
    main(args)
