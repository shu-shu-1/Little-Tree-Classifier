"""Export a wallpaper classifier checkpoint to ONNX and check numerical parity.

Examples
--------
python export_onnx.py --checkpoint checkpoints/best.pt
python export_onnx.py --checkpoint checkpoints/best.pt --output models/wallpaper.onnx --dynamic-batch

The script intentionally keeps preprocessing metadata next to the ONNX file;
ONNX contains the neural network only, not image decoding or normalization.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model import IMAGENET_MEAN, IMAGENET_STD, load_checkpoint


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/best.pt", help="PyTorch checkpoint")
    p.add_argument("--output", default=None, help="ONNX output path (default: checkpoint stem + .onnx)")
    p.add_argument("--metadata-output", default=None, help="Metadata JSON path (default: ONNX path + .json)")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset (minimum 17, default 17)")
    p.add_argument("--dynamic-batch", action="store_true", help="Allow a variable batch dimension")
    p.add_argument("--verify", dest="verify", action="store_true", default=True,
                   help="Run ONNX Runtime parity check when onnxruntime is installed (default)")
    p.add_argument("--no-verify", dest="verify", action="store_false", help="Skip ONNX Runtime parity check")
    p.add_argument("--device", default="cpu", help="Torch device used for export (default: cpu)")
    return p


def _require_onnx() -> Any:
    try:
        import onnx  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "缺少 ONNX 依赖，无法导出。请先运行：python -m pip install onnx "
            "onnxruntime（若只导出可省略 onnxruntime）"
        ) from exc
    return onnx


def _export(model: torch.nn.Module, sample: torch.Tensor, output: Path,
            opset: int, dynamic_batch: bool) -> str:
    input_names = ["input"]
    output_names = ["logits"]
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    # The dynamo exporter is preferred on recent PyTorch.  Fall back to the
    # stable legacy exporter for installations without onnxscript or older APIs.
    try:
        torch.onnx.export(
            model, (sample,), str(output),
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=True,
        )
        return "dynamo"
    except (TypeError, ImportError, ModuleNotFoundError) as exc:
        print(f"dynamo exporter unavailable ({exc}); falling back to legacy exporter", file=sys.stderr)
    torch.onnx.export(
        model, (sample,), str(output),
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    return "legacy"


def _verify(model: torch.nn.Module, output: Path, sample: torch.Tensor,
            dynamic_batch: bool) -> dict[str, Any]:
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        return {"status": "skipped", "reason": "onnxruntime is not installed"}

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(output), providers=providers)
    input_name = session.get_inputs()[0].name
    expected = model(sample).detach().cpu().numpy()
    actual = session.run(None, {input_name: sample.detach().cpu().numpy()})[0]
    max_abs = float(np.max(np.abs(expected - actual)))
    max_rel = float(np.max(np.abs(expected - actual) / np.maximum(np.abs(expected), 1e-8)))
    result: dict[str, Any] = {
        "status": "passed" if max_abs <= 1e-4 else "failed",
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "torch_shape": list(expected.shape),
        "onnx_shape": list(actual.shape),
        "argmax_equal": bool(np.argmax(expected, axis=1).tolist() == np.argmax(actual, axis=1).tolist()),
    }
    if dynamic_batch:
        # Exercise a second batch size so the dynamic axes are actually tested.
        sample2 = torch.cat([sample, sample], dim=0)
        expected2 = model(sample2).detach().cpu().numpy()
        actual2 = session.run(None, {input_name: sample2.detach().cpu().numpy()})[0]
        result["dynamic_batch"] = {
            "status": "passed" if expected2.shape == actual2.shape and np.max(np.abs(expected2 - actual2)) <= 1e-4 else "failed",
            "shape": list(actual2.shape),
            "max_abs_error": float(np.max(np.abs(expected2 - actual2))),
        }
    return result


def main() -> int:
    # PyTorch's exporter may emit non-ASCII status symbols.  Windows consoles
    # often default to GBK, so replace unencodable diagnostics instead of
    # aborting an otherwise successful export.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = _parser().parse_args()
    if args.opset < 17:
        print("--opset must be >= 17", file=sys.stderr)
        return 2
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2
    output = Path(args.output) if args.output else checkpoint.with_suffix(".onnx")
    metadata_path = Path(args.metadata_output) if args.metadata_output else Path(str(output) + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        onnx = _require_onnx()
        device = torch.device(args.device)
        model, ckpt = load_checkpoint(checkpoint, device=device)
        image_size = int(ckpt.get("image_size", 224))
        sample = torch.randn(1, 3, image_size, image_size, device=device)
        model.eval()
        with torch.inference_mode():
            exporter = _export(model, sample, output, args.opset, args.dynamic_batch)
        onnx_model = onnx.load(str(output))
        onnx.checker.check_model(onnx_model)
        actual_opsets = [int(item.version) for item in onnx_model.opset_import]
        actual_opset = max(actual_opsets) if actual_opsets else args.opset
        verification = _verify(model, output, sample, args.dynamic_batch) if args.verify else {
            "status": "skipped", "reason": "--no-verify"
        }
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ONNX export failed: {exc}", file=sys.stderr)
        return 1

    metadata = {
        "format": "wallpaper-classifier-onnx-v1",
        "checkpoint": str(checkpoint),
        "onnx": str(output),
        "external_data": [str(p.name) for p in output.parent.glob(output.name + ".data")],
        "model": ckpt.get("model", "unknown"),
        "class_names": ckpt["class_names"],
        "image_size": image_size,
        "preprocessing": ckpt.get("preprocessing", "stretch"),
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "input": {"name": "input", "shape": ["batch" if args.dynamic_batch else 1, 3, image_size, image_size], "dtype": "float32"},
        "output": {"name": "logits", "shape": ["batch" if args.dynamic_batch else 1, len(ckpt["class_names"])], "dtype": "float32"},
        "requested_opset": args.opset,
        "opset": actual_opset,
        "dynamic_batch": args.dynamic_batch,
        "exporter": exporter,
        "verification": verification,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported: {output}")
    print(f"Metadata: {metadata_path}")
    print(f"Verification: {verification['status']}")
    if verification["status"] == "failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
