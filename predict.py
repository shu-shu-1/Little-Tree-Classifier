#!/usr/bin/env python3
"""Run inference on one image with a checkpoint produced by train.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from model import load_checkpoint, preprocess_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--threshold", "--min-confidence", dest="threshold", type=float, default=0.0,
        help="将最高概率低于此值的结果标记为 unknown（0 表示始终输出分类）",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold 必须在 0 到 1 之间")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    image_size = int(checkpoint.get("image_size", 160))
    image = Image.open(args.image).convert("RGB")
    with torch.inference_mode():
        logits = model(preprocess_image(image, image_size, preprocessing=checkpoint.get('preprocessing', 'stretch')).unsqueeze(0).to(device))
        probabilities = logits.softmax(dim=1)[0]
    values, indices = probabilities.topk(min(args.top_k, probabilities.numel()))
    # Keep the original top-k output format for compatibility.  When a
    # threshold is requested, expose an explicit unknown result if the model
    # is not sufficiently confident instead of forcing the top class label.
    if args.threshold > 0.0 and float(values[0]) < args.threshold:
        print(f"unknown\t{float(values[0]):.4f}\t最高概率低于阈值 {args.threshold:.2f}")
    for value, index in zip(values.tolist(), indices.tolist()):
        print(f"{checkpoint['class_names'][index]}\t{value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
