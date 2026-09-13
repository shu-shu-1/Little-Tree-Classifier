"""Models and image preprocessing for wallpaper classification."""
from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any
import random
import torch
from torch import nn
from PIL import Image
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PREPROCESSING_MODES = ("stretch", "center_crop")


def resize_shorter_side_crop(image: Image.Image, image_size: int,
                             random_crop: bool = False) -> Image.Image:
    """Resize while preserving aspect ratio, then crop a square.

    The shorter edge is scaled to ``image_size`` before cropping.  Inference
    uses a deterministic center crop; training may request a random crop for
    additional framing augmentation.  Keeping this operation shared by the
    training and inference paths avoids the aspect-ratio distortion caused by
    directly resizing every image to a square.
    """
    if image_size < 1:
        raise ValueError("image_size must be positive")
    image = image.convert("RGB")
    w, h = image.size
    if w < 1 or h < 1:
        raise ValueError("image has an empty dimension")
    scale = image_size / min(w, h)
    nw, nh = max(image_size, round(w * scale)), max(image_size, round(h * scale))
    if (nw, nh) != (w, h):
        image = image.resize((nw, nh), Image.Resampling.BILINEAR)
    max_left, max_top = nw - image_size, nh - image_size
    if random_crop:
        left = random.randint(0, max_left) if max_left else 0
        top = random.randint(0, max_top) if max_top else 0
    else:
        left, top = max_left // 2, max_top // 2
    return image.crop((left, top, left + image_size, top + image_size))


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if name == "small-cnn":
        return SmallCNN(num_classes)
    if name in ("resnet18", "resnet18-feature", "resnet50", "resnet50-feature"):
        try:
            from torchvision import models
        except Exception as exc:
            raise RuntimeError("ResNet requires a working torchvision installation") from exc
        if name.startswith("resnet50"):
            ctor, enum = models.resnet50, models.ResNet50_Weights
        else:
            ctor, enum = models.resnet18, models.ResNet18_Weights
        weights = enum.DEFAULT if pretrained else None
        try:
            model = ctor(weights=weights)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    f"Could not load pretrained {name} weights. Check the network or "
                    "the torch weight cache; requested pretrained training cannot "
                    "continue with random weights."
                ) from exc
            raise
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if name.endswith("-feature"):
            for parameter in model.parameters():
                parameter.requires_grad = False
            for parameter in model.fc.parameters():
                parameter.requires_grad = True
            # Preserve the standard ResNet state_dict keys for old checkpoints.
            # requires_grad=False alone does not freeze BatchNorm running stats.
            model.train = MethodType(_train_frozen_features, model)
            model.train()
        return model
    raise ValueError(f"unknown model: {name}")


def _train_frozen_features(model: nn.Module, mode: bool = True) -> nn.Module:
    """Train the head while keeping every frozen backbone module in eval mode."""
    nn.Module.train(model, False)
    model.training = mode
    model.fc.train(mode)
    return model


def preprocess_image(
    image: Image.Image | str | Path,
    image_size: int = 224,
    preprocessing: str = "stretch",
) -> torch.Tensor:
    """Normalize an image, using the geometry recorded by its checkpoint.

    ``stretch`` is the legacy default. New training explicitly records
    ``center_crop`` so loading an older model never silently changes its input.
    """
    if image_size < 1:
        raise ValueError("image_size must be positive")
    if preprocessing not in PREPROCESSING_MODES:
        raise ValueError(f"unknown preprocessing: {preprocessing!r}")
    if not isinstance(image, Image.Image):
        with Image.open(image) as opened:
            return preprocess_image(opened, image_size, preprocessing)
    if preprocessing == "center_crop":
        image = resize_shorter_side_crop(image, image_size, random_crop=False)
    else:
        image = image.convert("RGB").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1))


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older PyTorch
        ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt.get("model", "small-cnn"), len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


@torch.inference_mode()
def predict_image(model: nn.Module, image: Image.Image | str | Path, checkpoint: dict[str, Any], device="cpu"):
    x = preprocess_image(
        image,
        int(checkpoint.get("image_size", 128)),
        preprocessing=checkpoint.get("preprocessing", "stretch"),
    ).unsqueeze(0).to(device)
    probs = model(x).softmax(1)[0]
    idx = int(probs.argmax())
    return checkpoint["class_names"][idx], float(probs[idx]), probs.cpu().tolist()
