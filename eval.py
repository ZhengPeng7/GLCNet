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

from copy import deepcopy
from tqdm import tqdm

from datasets import build_test_loader
from models.glcnet import GLCNet
from utils.utils import mkdir, load_weights, set_random_seed
from configs import config
from eval_func import eval_detection, eval_search_cuhk, eval_search_prw, eval_search_mvn


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


@torch.no_grad()
def evaluate_performance(
    model, gallery_loader, query_loader, device, use_gt=False, use_cache=False, use_cbgm=False, save_cache=False
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
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
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
                # Convert to float32 before numpy for consistent precision with bf16/fp16
                gallery_dets.append(box_w_scores.float().cpu().numpy())
                gallery_feats.append(output["embeddings"].float().cpu().numpy())

        # regarding query image as gallery to detect all people
        # i.e. query person + surrounding people (context information)
        query_dets, query_feats = [], []
        if use_cbgm:
            print(f'Extracting query context at {time.asctime(time.gmtime())} ...')
            for images, targets in tqdm(query_loader, ncols=0):
                # Move data to device
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
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
                    # Convert to float32 before numpy for consistent precision with bf16/fp16
                    query_dets.append(box_w_scores.float().cpu().numpy())
                    query_feats.append(output["embeddings"].float().cpu().numpy())

        # extract the features of query boxes
        query_box_feats = []
        print(f'Extracting query at {time.asctime(time.gmtime())} ...')
        for images, targets in tqdm(query_loader, ncols=0):
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            with accelerator.autocast():
                embeddings = model(images, targets, query_img_as_gallery=False)
            assert len(embeddings) == 1, "batch size in test phase should be 1"
            # Convert to float32 before numpy for consistent precision with bf16/fp16
            query_box_feats.append(embeddings[0].float().cpu().numpy())
        print(f'Finish feature extraction on gallery+query at {time.asctime(time.gmtime())} .')

        if save_cache:
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
    device = torch.device(config.device)
    if config.seed >= 0:
        set_random_seed(config.seed)

    print("Creating model")
    model = GLCNet(config)
    model.to(device)

    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')

    assert args.ckpt, "--ckpt must be specified for evaluation"
    load_weights(args.ckpt, model)

    print("Loading data")
    gallery_loader, query_loader = build_test_loader(config)

    # Handle --cbgm flag
    use_cbgm = args.cbgm or config.eval_use_cbgm
    print('use_cbgm:', use_cbgm)

    mAP, top1 = evaluate_performance(
        model,
        gallery_loader,
        query_loader,
        device,
        use_gt=config.eval_use_gt,
        use_cache=args.use_cache,
        use_cbgm=use_cbgm,
        save_cache=args.save_cache,
    )
    print(f"Evaluation results: mAP={mAP:.4f}, top-1={top1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a person search network.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint to evaluate.")
    parser.add_argument("--cbgm", action="store_true", help="Use CBGM algorithm for evaluation.")
    parser.add_argument("--use_cache", action="store_true", help="Use cached features for evaluation.")
    parser.add_argument("--save_cache", action="store_true", help="Save extracted features to cache.")
    args = parser.parse_args()
    main(args)
