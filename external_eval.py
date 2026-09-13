#!/usr/bin/env python3
"""Evaluate a checkpoint on an independently collected, folder-labeled image tree.

The evaluator never copies or modifies the external files.  A folder is treated as
the label when its name matches ``--map`` patterns (JSON object, pattern -> class),
or when ``--class-dir`` points directly at one labeled directory.  Unmatched files
are skipped, which is useful for a mixed personal photo archive.
"""
from __future__ import annotations

import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path
import torch
from PIL import Image
from model import load_checkpoint, preprocess_image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".avif", ".jfif"}

# Conservative defaults: only clearly named character/collection folders are
# considered anime.  Add mappings with --map-file for your own directory names.
DEFAULT_PATTERNS = {
    "八重神子": "anime", "白圣女与黑牧师": "anime", "冰之女皇": "anime",
    "芙宁娜": "anime", "灵砂": "anime", "麻衣学姐": "anime", "千织": "anime",
    "阮·梅": "anime", "丝柯克": "anime", "散兵": "anime", "那维莱特": "anime",
    "穗": "anime", "温迪": "anime", "遐蝶": "anime", "西安游": "landscape",
    "风景": "landscape", "宠物": "pets", "猫": "pets", "狗": "pets",
    "汽车": "cars", "车辆": "cars", "城市": "city", "太空": "space",
}

def load_patterns(path: Path | None) -> dict[str, str]:
    if not path:
        return DEFAULT_PATTERNS
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("map file must contain a JSON object: folder substring -> class")
    return {str(k): str(v) for k, v in obj.items()}

def label_for(path: Path, root: Path, patterns: dict[str, str], direct: dict[str, str]) -> str | None:
    rel = path.relative_to(root)
    # Direct class directories are exact first-level labels.
    if rel.parts and rel.parts[0] in direct:
        return direct[rel.parts[0]]
    # Match any ancestor folder by substring, longest pattern wins.
    candidates = []
    for part in rel.parts[:-1]:
        for pat, cls in patterns.items():
            if pat in part:
                candidates.append((len(pat), cls))
    return max(candidates, default=(0, None))[1]

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="external image directory")
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    ap.add_argument("--map-file", type=Path, help="JSON mapping of folder substring to model class")
    ap.add_argument("--class-dir", action="append", metavar="CLASS=DIR",
                    help="explicit labeled directory; repeatable, e.g. anime=八重神子")
    ap.add_argument("--output", type=Path, default=Path("checkpoints/external_eval.csv"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir(): ap.error(f"not a directory: {root}")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    model, ck = load_checkpoint(args.checkpoint, device)
    classes = list(ck["class_names"])
    patterns = load_patterns(args.map_file)
    direct = {}
    for item in args.class_dir or []:
        if "=" not in item: ap.error("--class-dir requires CLASS=DIR")
        cls, folder = item.split("=", 1); direct[folder] = cls
    rows, counts, correct = [], Counter(), Counter()
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in EXTS: continue
        label = label_for(p, root, patterns, direct)
        if label not in classes: continue
        try: image = Image.open(p).convert("RGB")
        except Exception: continue
        with torch.inference_mode():
            prob = model(preprocess_image(image, int(ck.get("image_size", 128)), preprocessing=ck.get('preprocessing', 'stretch')).unsqueeze(0).to(device)).softmax(1)[0]
        idx = int(prob.argmax()); pred, conf = classes[idx], float(prob[idx])
        counts[label] += 1; correct[label] += int(pred == label)
        rows.append((str(p), label, pred, conf))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["path", "label", "prediction", "confidence"]); w.writerows(rows)
    print(json.dumps({"root": str(root), "evaluated": len(rows), "accuracy": sum(correct.values()) / len(rows) if rows else None,
                      "per_class": {c: {"count": counts[c], "correct": correct[c], "accuracy": correct[c] / counts[c] if counts[c] else None} for c in sorted(counts)}}, ensure_ascii=False, indent=2))
    print(f"details written to {args.output}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
