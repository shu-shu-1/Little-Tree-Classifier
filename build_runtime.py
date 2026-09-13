"""Build a versioned standalone classifier-runtime package.

Examples:
    python build_runtime.py --checkpoint checkpoints/best.pt --version 1.0.0
    python build_runtime.py --checkpoint checkpoints/balanced.pt --version 1.0.0 --debug-dir
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_onnx(checkpoint_path: Path, output: Path) -> dict:
    import torch

    sys.path.insert(0, str(ROOT))
    from model import load_checkpoint

    model, checkpoint = load_checkpoint(checkpoint_path, "cpu")
    image_size = int(checkpoint.get("image_size", 160))
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 3, image_size, image_size)
    try:
        # PyTorch 2.9+ recommends the torch.export-based exporter. It also
        # avoids the deprecation warning emitted by the legacy exporter.
        # Capture exporter progress because it may contain Unicode symbols
        # that cannot be written by a Windows GBK console.
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            torch.onnx.export(
                model,
                (example,),
                output,
                input_names=["images"],
                output_names=["logits"],
                dynamic_shapes=({0: "batch"},),
                opset_version=18,
                dynamo=True,
            )
    except Exception as exc:
        # Keep releases usable with older PyTorch versions or models that the
        # new exporter cannot lower yet. The fallback is intentionally narrow
        # and reports why the modern path was not used.
        print(f"warning: modern ONNX exporter failed; using legacy exporter: {exc!r}", file=sys.stderr)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX export.*")
            torch.onnx.export(
                model,
                example,
                output,
                input_names=["images"],
                output_names=["logits"],
                dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best.pt")
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "release")
    parser.add_argument("--debug-dir", action="store_true", help="also keep an unpacked runtime directory")
    parser.add_argument("--skip-pyinstaller", action="store_true", help="create the source runtime only")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path}")

    work = args.output.resolve() / f"classifier-runtime-v{args.version}-windows-x64.work"
    runtime = args.output.resolve() / f"classifier-runtime-v{args.version}-windows-x64"
    archive = args.output.resolve() / f"classifier-runtime-v{args.version}-windows-x64.zip"
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(runtime, ignore_errors=True)
    archive.unlink(missing_ok=True)
    work.mkdir(parents=True)

    checkpoint = export_onnx(checkpoint_path, work / "model.onnx")
    (work / "labels.json").write_text(json.dumps(checkpoint["class_names"], ensure_ascii=False, indent=2), encoding="utf-8")
    preprocessing = {
        "image_size": int(checkpoint.get("image_size", 160)),
        "mode": checkpoint.get("preprocessing") or "stretch",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    (work / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    shutil.copy2(ROOT / "classifier_worker.py", work / "classifier_worker.py")

    model_hash = sha256(work / "model.onnx")
    if not args.skip_pyinstaller:
        if shutil.which("pyinstaller") is None:
            raise SystemExit("PyInstaller is not installed. Run: python -m pip install pyinstaller")
        spec = work / "classifier_worker.spec"
        spec.write_text(
            """from PyInstaller.utils.hooks import collect_submodules\n\nhiddenimports = collect_submodules('onnxruntime')\na = Analysis(['classifier_worker.py'], pathex=['.'], binaries=[], datas=[('model.onnx', '.'), ('labels.json', '.'), ('preprocessing.json', '.')], hiddenimports=hiddenimports)\npyz = PYZ(a.pure)\nexe = EXE(pyz, a.scripts, a.binaries, a.datas, name='classifier-worker', console=True)\n""",
            encoding="utf-8",
        )
        subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(spec), "--distpath", str(work)], check=True)
        built = work / "classifier-worker.exe"
        if not built.is_file():
            raise SystemExit(f"PyInstaller did not create {built}")
        for path in (work / "model.onnx", work / "labels.json", work / "preprocessing.json"):
            path.unlink(missing_ok=True)
    else:
        (work / "classifier-worker.py").unlink(missing_ok=True)
        (work / "classifier_worker.py").replace(work / "classifier-worker.py")

    manifest = {
        "id": "wallpaper-classifier",
        "version": args.version,
        "protocol_version": 1,
        "platform": "windows",
        "architecture": "x64",
        "entry": "classifier-worker.exe" if not args.skip_pyinstaller else "classifier-worker.py",
        "model": "model.onnx",
        "checkpoint": checkpoint_path.name,
        "labels": checkpoint["class_names"],
        "model_sha256": model_hash,
    }
    (work / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in work.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(work))
    if args.debug_dir:
        shutil.copytree(work, runtime)
    shutil.rmtree(work)
    print(f"created: {archive}")
    print(f"sha256: {sha256(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
