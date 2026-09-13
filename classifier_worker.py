"""JSON-lines ONNX inference worker for the standalone classifier runtime."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _runtime_dir() -> Path:
    # PyInstaller sets _MEIPASS for one-file extraction; folder builds use the
    # executable directory. Source execution uses this file's directory.
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return Path(bundled)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


class Classifier:
    def __init__(self, root: Path):
        import onnxruntime as ort

        self.root = root
        self.labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
        self.preprocessing = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
        self.session = ort.InferenceSession(
            str(root / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def _input(self, image_path: Path) -> np.ndarray:
        size = int(self.preprocessing["image_size"])
        mode = self.preprocessing.get("mode", "stretch")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            if mode == "center_crop":
                width, height = image.size
                scale = size / min(width, height)
                dimensions = (max(size, round(width * scale)), max(size, round(height * scale)))
                image = image.resize(dimensions, Image.Resampling.BILINEAR)
                left = (image.width - size) // 2
                top = (image.height - size) // 2
                image = image.crop((left, top, left + size, top + size))
            else:
                image = image.resize((size, size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0

        mean = np.asarray(self.preprocessing["mean"], dtype=np.float32)
        std = np.asarray(self.preprocessing["std"], dtype=np.float32)
        array = (array - mean) / std
        return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...], dtype=np.float32)

    def classify(self, image_path: str, top_k: int = 3) -> list[dict[str, Any]]:
        values = self.session.run(None, {self.input_name: self._input(Path(image_path))})[0][0]
        values = values.astype(np.float64)
        values -= np.max(values)
        probabilities = np.exp(values)
        probabilities /= np.sum(probabilities)
        indices = np.argsort(probabilities)[::-1][: max(1, min(top_k, len(self.labels)))]
        return [
            {"label": self.labels[int(index)], "score": float(probabilities[index]), "index": int(index)}
            for index in indices
        ]


def _reply(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    try:
        classifier = Classifier(_runtime_dir())
    except Exception as exc:
        _reply({"ok": False, "error": f"initialization failed: {exc}"})
        return 2

    _reply({"ok": True, "action": "ready", "labels": classifier.labels})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            action = request.get("action")
            if action == "health":
                _reply({"ok": True, "action": "health"})
            elif action == "classify":
                _reply({
                    "ok": True,
                    "action": "classify",
                    "results": classifier.classify(str(request["image_path"]), int(request.get("top_k", 3))),
                })
            elif action == "shutdown":
                _reply({"ok": True, "action": "shutdown"})
                return 0
            else:
                _reply({"ok": False, "error": f"unknown action: {action!r}"})
        except Exception as exc:
            _reply({"ok": False, "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
