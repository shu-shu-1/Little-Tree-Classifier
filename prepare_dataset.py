#!/usr/bin/env python3
"""Validate downloaded images and create deterministic train/val/test folders."""
from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def valid_image(path: Path, min_width: int = 128, min_height: int = 128) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.width >= min_width and image.height >= min_height
    except Exception:
        return False


def split_files(files: list[Path], seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    files = sorted(files, key=lambda p: hashlib.sha1(str(p).encode()).hexdigest())
    random.Random(seed).shuffle(files)
    n = len(files)
    n_test = max(1, round(n * 0.15)) if n >= 3 else 0
    n_val = max(1, round(n * 0.15)) if n >= 5 else 0
    return files[n_test + n_val:], files[n_test:n_test + n_val], files[:n_test]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/split"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-images", type=int, default=5)
    parser.add_argument("--min-width", type=int, default=256,
                        help="discard images narrower than this (default: 256)")
    parser.add_argument("--min-height", type=int, default=256,
                        help="discard images shorter than this (default: 256)")
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="cap each class before splitting; 0 keeps all (useful for severe imbalance)")
    parser.add_argument("--deduplicate", action="store_true",
                        help="remove exact and near-duplicate images globally (perceptual hash)")
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"input directory not found: {args.input}; run download_dataset.py first")
    if args.input.resolve() == args.output.resolve() or args.input.resolve() in args.output.resolve().parents:
        raise SystemExit("output must be outside the raw input directory")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output}; choose a new --output directory")
    summary: dict[str, dict[str, int]] = {}
    # Keep one representative for near-identical images.  This is intentionally
    # non-destructive: source files remain in data/raw; only the generated split
    # omits redundant copies.
    seen_hashes: dict[str, Path] = {}
    for class_dir in sorted(p for p in args.input.iterdir() if p.is_dir()):
        files = [p for p in class_dir.rglob("*") if p.is_file()
                 and valid_image(p, args.min_width, args.min_height)]
        if args.deduplicate:
            unique = []
            for p in sorted(files):
                try:
                    with Image.open(p) as im:
                        # 8x8 average hash catches resized/re-encoded duplicates.
                        g = im.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
                        px = list(g.getdata()); avg = sum(px) / len(px)
                        bits = "".join("1" if v >= avg else "0" for v in px)
                        key = f"{int(bits, 2):016x}"
                except Exception:
                    continue
                # Hamming distance <= 2 is close enough to be a duplicate for
                # wallpaper downloads while retaining visually distinct images.
                duplicate = False
                for old in seen_hashes:
                    if (int(key, 16) ^ int(old, 16)).bit_count() <= 2:
                        duplicate = True; break
                if not duplicate:
                    seen_hashes[key] = p; unique.append(p)
            files = unique
        if args.max_per_class and len(files) > args.max_per_class:
            # Deterministic, spread-out selection avoids retaining only one
            # scraper/source's contiguous filenames.
            ordered = sorted(files, key=lambda p: hashlib.sha1(str(p).encode()).hexdigest())
            files = ordered[:args.max_per_class]
        if len(files) < args.min_images:
            print(f"skip {class_dir.name}: only {len(files)} valid images")
            continue
        splits = split_files(files, args.seed)
        summary[class_dir.name] = {}
        for split_name, split_files_list in zip(("train", "val", "test"), splits):
            destination = args.output / split_name / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(split_files_list):
                # Re-encode through Pillow to normalize formats and avoid broken files.
                target = destination / f"{index:05d}_{source.stem}.jpg"
                try:
                    with Image.open(source) as image:
                        image.convert("RGB").save(target, "JPEG", quality=92)
                except Exception:
                    target.unlink(missing_ok=True)
            summary[class_dir.name][split_name] = len(list(destination.glob("*.jpg")))
    print("Prepared dataset:")
    for category, counts in summary.items():
        print(f"  {category}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    if not summary:
        raise SystemExit("no usable classes found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
