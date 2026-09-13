#!/usr/bin/env python3
"""Tkinter test interface for wallpaper image classification.

Run ``python gui_app.py`` from the project directory.  A checkpoint can be
selected on the command line with ``--checkpoint`` (defaults to
``checkpoints/best.pt``).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch
from PIL import Image, ImageTk

from model import load_checkpoint, predict_image


class WallpaperClassifierApp:
    """Small desktop UI around :func:`model.predict_image`."""

    PREVIEW_SIZE = (520, 360)

    def __init__(self, root: tk.Tk, checkpoint_path: Path, top_k: int = 5, device: str = "auto", confidence_threshold: float = 0.0):
        self.root = root
        self.root.title("壁纸图片分类测试")
        self.root.minsize(680, 520)
        self.top_k = max(1, int(top_k))
        if not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold 必须在 0 到 1 之间")
        self.device = self._resolve_device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self.checkpoint = None
        self.current_path: Path | None = None
        self.current_image: Image.Image | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None

        self.path_var = tk.StringVar(value="尚未选择图片")
        self.device_var = tk.StringVar(value=f"设备：{self.device}")
        self.result_var = tk.StringVar(value="请选择一张图片开始测试")
        self.speed_var = tk.StringVar(value="推理速度：等待预测")
        self.threshold_var = tk.DoubleVar(value=float(confidence_threshold))
        self.threshold_text_var = tk.StringVar(value="低置信度提示阈值：关闭")

        self._build_widgets()
        self._load_model()

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested

    def _build_widgets(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(toolbar, text="选择图片", command=self.choose_image).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="开始预测", command=self.predict_current).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(toolbar, textvariable=self.device_var).pack(side=tk.RIGHT)

        threshold_frame = ttk.Frame(outer)
        threshold_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(threshold_frame, text="低置信度阈值：").pack(side=tk.LEFT)
        threshold = ttk.Scale(threshold_frame, from_=0.0, to=1.0, variable=self.threshold_var,
                              command=self._on_threshold_change)
        threshold.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Label(threshold_frame, textvariable=self.threshold_text_var, width=18).pack(side=tk.RIGHT)

        path_frame = ttk.Frame(outer)
        path_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(path_frame, text="图片：").pack(side=tk.LEFT)
        ttk.Label(path_frame, textvariable=self.path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        content = ttk.Frame(outer)
        content.pack(fill=tk.BOTH, expand=True)
        preview_box = ttk.LabelFrame(content, text="预览")
        preview_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self.preview_label = ttk.Label(preview_box, text="选择图片后显示预览", anchor=tk.CENTER)
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        result_box = ttk.LabelFrame(content, text="分类结果（Top-K）", width=230)
        result_box.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(8, 0))
        result_box.pack_propagate(False)
        ttk.Label(result_box, textvariable=self.result_var, wraplength=210, justify=tk.LEFT).pack(
            anchor=tk.W, padx=10, pady=(10, 6)
        )
        ttk.Label(result_box, textvariable=self.speed_var, wraplength=210, justify=tk.LEFT).pack(
            anchor=tk.W, padx=10, pady=(0, 6)
        )
        self.results_text = tk.Text(result_box, width=28, height=16, state=tk.DISABLED, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        ttk.Label(outer, text=f"模型：{self.checkpoint_path}", foreground="#666").pack(anchor=tk.W, pady=(8, 0))

    def _load_model(self) -> None:
        try:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"找不到模型文件：{self.checkpoint_path}")
            self.model, self.checkpoint = load_checkpoint(self.checkpoint_path, self.device)
            classes = self.checkpoint.get("class_names", [])
            self.result_var.set(f"模型已加载，共 {len(classes)} 个分类。请选择图片")
        except Exception as exc:
            self.model = None
            self.checkpoint = None
            messagebox.showerror("模型加载失败", str(exc), parent=self.root)
            self.result_var.set("模型加载失败，请检查 checkpoint 路径")

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择壁纸图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp *.gif"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        try:
            # Load and copy so the file handle is closed while the UI is open.
            with Image.open(path) as image:
                self.current_image = image.convert("RGB")
            self.current_path = Path(path)
            self.path_var.set(str(self.current_path))
            self._show_preview(self.current_image)
            self.predict_current()  # auto-predict after selecting an image
        except Exception as exc:
            self.current_image = None
            self.current_path = None
            messagebox.showerror("图片打开失败", str(exc), parent=self.root)

    def _show_preview(self, image: Image.Image) -> None:
        preview = image.copy()
        preview.thumbnail(self.PREVIEW_SIZE, Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self.preview_photo, text="")

    def _set_results_text(self, lines: list[str]) -> None:
        self.results_text.configure(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "\n".join(lines))
        self.results_text.configure(state=tk.DISABLED)

    def _on_threshold_change(self, _value: str = "") -> None:
        value = float(self.threshold_var.get())
        self.threshold_text_var.set("关闭" if value < 0.01 else f"{value:.0%} 以下提示未知")

    def predict_current(self) -> None:
        if self.model is None or self.checkpoint is None:
            messagebox.showerror("无法预测", "模型尚未成功加载。", parent=self.root)
            return
        if self.current_image is None:
            messagebox.showinfo("提示", "请先选择一张图片。", parent=self.root)
            return
        try:
            # CUDA launches are asynchronous; synchronize around the timer so
            # the displayed duration reflects the complete inference call.
            use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
            if use_cuda:
                torch.cuda.synchronize()
            started = time.perf_counter()
            predicted, confidence, probabilities = predict_image(
                self.model, self.current_image, self.checkpoint, self.device
            )
            if use_cuda:
                torch.cuda.synchronize()
            elapsed = max(time.perf_counter() - started, 1e-9)
            fps = 1.0 / elapsed
            class_names = self.checkpoint["class_names"]
            pairs = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
            pairs = pairs[: min(self.top_k, len(pairs))]
            lines = [f"{rank}. {class_names[index]}    {prob * 100:.2f}%" for rank, (index, prob) in enumerate(pairs, 1)]
            threshold = float(self.threshold_var.get())
            shown_label = "未知/信心不足" if threshold >= 0.01 and confidence < threshold else predicted
            threshold_hint = (
                f"，阈值 {threshold * 100:.0f}%"
                if threshold >= 0.01 else ""
            )
            self.result_var.set(
                f"预测：{shown_label}（最高置信度 {confidence * 100:.2f}%{threshold_hint}）"
            )
            self.speed_var.set(f"推理耗时：{elapsed * 1000:.2f} ms\n速度：{fps:.2f} FPS")
            self._set_results_text(lines)
        except Exception as exc:
            self.speed_var.set("推理速度：预测失败")
            messagebox.showerror("预测失败", str(exc), parent=self.root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"), help="模型 checkpoint 路径")
    parser.add_argument("--top-k", type=int, default=5, help="显示概率最高的类别数量")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="推理设备")
    parser.add_argument(
        "--threshold", "--min-confidence", dest="threshold", type=float, default=0.0,
        help="最高概率低于此值时显示 unknown（0 表示始终显示分类）",
    )
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold 必须在 0 到 1 之间")

    root = tk.Tk()
    WallpaperClassifierApp(root, args.checkpoint, args.top_k, args.device, args.threshold)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
