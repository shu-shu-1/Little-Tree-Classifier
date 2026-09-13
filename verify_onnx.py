#!/usr/bin/env python3
"""Compare an exported ONNX model with its PyTorch checkpoint.

The check covers deterministic random tensors and real images from a dataset
split.  It reports numerical differences for logits/probabilities and top-1
agreement in JSON, making it suitable for a quick post-export smoke test.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from model import load_checkpoint, preprocess_image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _stats(torch_logits: np.ndarray, onnx_logits: np.ndarray) -> dict[str, Any]:
    """Return numerical and classification agreement statistics."""
    a = np.asarray(torch_logits, dtype=np.float32)
    b = np.asarray(onnx_logits, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"output shape mismatch: PyTorch {a.shape}, ONNX {b.shape}")
    diff = np.abs(a - b)
    pa = torch.softmax(torch.from_numpy(a), dim=-1).numpy()
    pb = torch.softmax(torch.from_numpy(b), dim=-1).numpy()
    pdiff = np.abs(pa - pb)
    ta, tb = a.argmax(axis=-1), b.argmax(axis=-1)
    return {
        "samples": int(a.shape[0]),
        "logits_max_abs_diff": float(diff.max(initial=0.0)),
        "logits_mean_abs_diff": float(diff.mean()),
        "probabilities_max_abs_diff": float(pdiff.max(initial=0.0)),
        "probabilities_mean_abs_diff": float(pdiff.mean()),
        "top1_agreement": float((ta == tb).mean()) if len(ta) else 1.0,
        "top1_agree_count": int((ta == tb).sum()),
    }


def _load_onnx(path: Path):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for verification. Install it with "
            "`python -m pip install onnxruntime` (or onnxruntime-gpu)."
        ) from exc
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), providers=providers)
    if not session.get_inputs():
        raise RuntimeError("ONNX model has no inputs")
    if not session.get_outputs():
        raise RuntimeError("ONNX model has no outputs")
    return session


def _real_images(root: Path, split: str, limit: int) -> list[Path]:
    base = root / split
    if not base.exists():
        return []
    paths = sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    return paths[: max(0, limit)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--onnx", type=Path, required=True, help="Path to exported .onnx model")
    parser.add_argument("--data", type=Path, default=None, help="Dataset root containing split folders")
    parser.add_argument("--split", default="test", help="Dataset split for real-image checks (default: test)")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--random-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--report", type=Path, default=Path("reports/onnx_verification.json"))
    parser.add_argument("--atol", type=float, default=1e-4, help="Maximum allowed absolute logit difference")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance (informational)")
    args = parser.parse_args()
    if args.random_samples < 0 or args.max_images < 0:
        parser.error("--random-samples and --max-images must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        model, checkpoint = load_checkpoint(args.checkpoint, "cpu")
        session = _load_onnx(args.onnx)
        image_size = int(checkpoint.get("image_size", 224))
        preprocessing = checkpoint.get("preprocessing", "stretch")
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name

        torch_batches: list[np.ndarray] = []
        onnx_batches: list[np.ndarray] = []
        if args.random_samples:
            # Evaluate one tensor at a time so exports with a static batch
            # dimension of 1 are supported as well as dynamic-batch models.
            for _ in range(args.random_samples):
                x = torch.randn((1, 3, image_size, image_size), dtype=torch.float32)
                with torch.inference_mode():
                    y = model(x).cpu().numpy()
                y_onnx = session.run(None, {input_name: x.numpy()})[0]
                torch_batches.append(y)
                onnx_batches.append(np.asarray(y_onnx))

        data_root = args.data
        if data_root is None:
            data_root = Path(checkpoint.get("config", {}).get("data", "data/split"))
        image_paths = _real_images(data_root, args.split, args.max_images)
        real_paths: list[str] = []
        for path in image_paths:
            try:
                with Image.open(path) as im:
                    x = preprocess_image(im, image_size, preprocessing).unsqueeze(0)
                with torch.inference_mode():
                    y = model(x).cpu().numpy()
                y_onnx = session.run(None, {input_name: x.numpy()})[0]
                torch_batches.append(y)
                onnx_batches.append(np.asarray(y_onnx))
                real_paths.append(str(path))
            except Exception as exc:
                print(f"warning: skipped {path}: {exc}", file=sys.stderr)

        if not torch_batches:
            raise RuntimeError("No samples available; increase --random-samples or provide --data/--split")
        all_torch = np.concatenate(torch_batches, axis=0)
        all_onnx = np.concatenate(onnx_batches, axis=0)
        random_count = args.random_samples
        report = {
            "checkpoint": str(args.checkpoint),
            "onnx": str(args.onnx),
            "input": {"name": input_name, "shape": input_meta.shape, "image_size": image_size, "preprocessing": preprocessing},
            "random": _stats(all_torch[:random_count], all_onnx[:random_count]) if random_count else {"samples": 0},
            "real": _stats(all_torch[random_count:], all_onnx[random_count:]),
            "overall": _stats(all_torch, all_onnx),
            "real_images": real_paths,
            "tolerances": {"atol": args.atol, "rtol": args.rtol},
        }
        report["passed"] = bool(
            report["overall"]["logits_max_abs_diff"] <= args.atol
            and report["overall"]["top1_agreement"] == 1.0
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("random", "real", "overall", "passed")}, ensure_ascii=False, indent=2))
        print(f"Report written to {args.report}")
        return 0 if report["passed"] else 2
    except Exception as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
