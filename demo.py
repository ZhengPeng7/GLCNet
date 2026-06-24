import argparse
import os
from glob import glob

import matplotlib
matplotlib.use("Agg")  # headless rendering for batch inference (no display needed)
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from PIL import Image
from torchvision.transforms import functional as F

from configs import config
from models.glcnet import GLCNet
from utils.utils import load_weights


def visualize_result(img_path, detections, similarities, save_path):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(plt.imread(img_path))
    plt.axis("off")
    for detection, sim in zip(detections, similarities):
        x1, y1, x2, y2 = detection
        ax.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#4CAF50", linewidth=3.5
            )
        )
        ax.add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=1)
        )
        ax.text(
            x1 + 5,
            y1 - 18,
            "{:.2f}".format(sim),
            bbox=dict(facecolor="#4CAF50", linewidth=0),
            fontsize=20,
            color="white",
        )
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _gallery_paths(gallery):
    """Resolve --gallery to a sorted list of image paths: accepts a directory or a glob pattern."""
    if os.path.isdir(gallery):
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths += glob(os.path.join(gallery, ext))
        return sorted(paths)
    return sorted(glob(gallery))


def main(args):
    device = torch.device(config.device)

    print("Creating model")
    model = GLCNet()
    model.to(device)
    model.eval()

    load_weights(args.ckpt, model)

    qbox = [float(v) for v in args.query_box.split(",")]
    query_img = [F.to_tensor(Image.open(args.query).convert("RGB")).to(device)]
    query_target = [{"boxes": torch.tensor([qbox]).to(device)}]
    query_feat = model(query_img, query_target)[0]

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    gallery_img_paths = _gallery_paths(args.gallery)
    for gallery_img_path in gallery_img_paths:
        print(f"Processing {gallery_img_path}")
        gallery_img = [F.to_tensor(Image.open(gallery_img_path).convert("RGB")).to(device)]
        gallery_output = model(gallery_img)[0]
        detections = gallery_output["boxes"]
        gallery_feats = gallery_output["embeddings"]

        # Compute pairwise cosine similarities in float32 for numerical stability
        # which equals to inner-products, as features are already L2-normed.
        # squeeze(1) (not squeeze()) keeps the per-detection dim when only 1 person is detected.
        similarities = gallery_feats.float().mm(query_feat.float().view(-1, 1)).squeeze(1)

        if args.out_dir:
            save_path = os.path.join(args.out_dir, os.path.basename(gallery_img_path))
        else:
            save_path = gallery_img_path.replace("gallery", "result")  # bundled-demo default
        visualize_result(gallery_img_path, detections, similarities, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLCNet person-search inference / visualization.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint.")
    parser.add_argument("--query", default="demo_imgs/query.jpg", help="Query image path.")
    parser.add_argument("--query_box", default="0,0,466,943", help="Query box as 'x1,y1,x2,y2'.")
    parser.add_argument("--gallery", default="demo_imgs/gallery-*.jpg", help="Gallery: a directory or a glob pattern.")
    parser.add_argument("--out_dir", default=None, help="Output dir for result images; default writes <result-*> next to the bundled gallery.")
    args = parser.parse_args()
    main(args)
