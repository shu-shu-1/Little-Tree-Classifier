#!/usr/bin/env python3
"""Evaluate a checkpoint on an unlabelled external image folder.

The script never copies or modifies source images. It reports predictions,
confidence and per-subfolder distributions, which is useful for spotting
domain shift or accidental train-set leakage in personal wallpaper folders.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image

from model import load_checkpoint, preprocess_image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".avif"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", type=Path, default=Path("checkpoints/external_predictions.csv"))
    ap.add_argument("--min-confidence", type=float, default=0.0)
    args = ap.parse_args()
    if not args.folder.exists():
        ap.error(f"folder not found: {args.folder}")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    size = int(ckpt.get("image_size", 128)); classes = ckpt["class_names"]
    files = sorted(p for p in args.folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []; grouped = defaultdict(Counter); skipped = 0
    for path in files:
        try:
            with Image.open(path) as im:
                image = im.convert("RGB")
            with torch.inference_mode():
                prob = model(preprocess_image(image, size, preprocessing=ckpt.get('preprocessing', 'stretch')).unsqueeze(0).to(device)).softmax(1)[0].cpu()
            value, idx = prob.max(0); label = classes[int(idx)]; conf = float(value)
            parent = path.parent.name
            grouped[parent][label] += 1
            rows.append({"file": str(path), "folder": parent, "prediction": label,
                         "confidence": f"{conf:.6f}", "sha256": sha256(path),
                         "low_confidence": conf < args.min_confidence})
        except Exception:
            skipped += 1
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "folder", "prediction", "confidence", "sha256", "low_confidence"])
        writer.writeheader(); writer.writerows(rows)
    print(f"evaluated {len(rows)} images; skipped {skipped}; wrote {args.output}")
    for folder, counts in sorted(grouped.items()):
        print(f"{folder}: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
