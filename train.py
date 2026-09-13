"""Train a wallpaper classifier on data/split/{train,val,test}/<class> folders."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from model import PREPROCESSING_MODES, build_model, preprocess_image, resize_shorter_side_crop

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SELECTION_METRIC = "val_macro_f1"


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root: Path,
        class_names: list[str],
        image_size: int,
        train: bool = False,
        preprocessing: str = "stretch",
    ):
        if preprocessing not in PREPROCESSING_MODES:
            raise ValueError(f"unknown preprocessing: {preprocessing!r}")
        self.image_size = image_size
        self.train = train
        self.preprocessing = preprocessing
        indices = {name: index for index, name in enumerate(class_names)}
        self.items = [
            (path, indices[path.parent.name])
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.suffix.lower() in EXTS
            and path.parent.name in indices
        ]
        if not self.items:
            raise ValueError(f"no images found under {root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        try:
            with Image.open(path) as opened:
                if self.train:
                    # Limit expensive rotations/color transforms on 4K/8K sources.
                    # JPEG draft also reduces decoder memory when supported.
                    maximum_side = max(512, self.image_size * 3)
                    opened.draft("RGB", (maximum_side, maximum_side))
                    opened.thumbnail(
                        (maximum_side, maximum_side), Image.Resampling.BILINEAR
                    )
                image = opened.convert("RGB")
        except Exception as exc:
            # Substituting another index silently changes labels and sampling
            # probabilities, and recursion never terminates for an invalid set.
            raise RuntimeError(f"cannot decode dataset image {path}: {exc}") from exc

        if self.train:
            image = self._augment(image)
            if self.preprocessing == "center_crop":
                image = resize_shorter_side_crop(
                    image, self.image_size, random_crop=True
                )
        return preprocess_image(image, self.image_size, self.preprocessing), label

    @staticmethod
    def _augment(image: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        width, height = image.size
        if min(width, height) > 8 and random.random() < 0.8:
            scale = random.uniform(0.82, 1.0)
            crop_width = max(8, int(width * scale))
            crop_height = max(8, int(height * scale))
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)
            image = image.crop((left, top, left + crop_width, top + crop_height))
        if random.random() < 0.25:
            image = image.rotate(
                random.uniform(-10, 10),
                resample=Image.Resampling.BILINEAR,
                expand=False,
            )
        image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        if random.random() < 0.12:
            image = image.filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.6))
            )
        return image


def evaluate(model, loader, device, nclasses):
    model.eval()
    total = correct = 0
    matrix = np.zeros((nclasses, nclasses), dtype=np.int64)
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            labels = labels.cpu()
            correct += int((predictions == labels).sum())
            total += labels.numel()
            for actual, predicted in zip(labels.tolist(), predictions.tolist()):
                matrix[actual, predicted] += 1
    return (correct / total if total else 0.0), matrix


def per_class_metrics(matrix, class_names):
    """Report support as well as precision/recall/F1, including absent classes."""
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(1).astype(np.int64)
    predicted = matrix.sum(0).astype(np.int64)
    recall = np.divide(
        true_positive, support, out=np.zeros_like(true_positive), where=support > 0
    )
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    return {
        name: {
            "support": int(support[index]),
            "predicted": int(predicted[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, name in enumerate(class_names)
    }


def classification_metrics(matrix):
    """Return macro-F1 and balanced accuracy over classes with ground truth."""
    rows = per_class_metrics(matrix, [str(index) for index in range(len(matrix))])
    present = [row for row in rows.values() if row["support"] > 0]
    if not present:
        return 0.0, 0.0
    return (
        float(np.mean([row["f1"] for row in present])),
        float(np.mean([row["recall"] for row in present])),
    )


def dataset_fingerprint(datasets, data_root: Path, class_names):
    """Hash labels, relative paths and file content so split changes are visible."""
    combined = hashlib.sha256()
    splits = {}
    for split, dataset in datasets.items():
        if dataset is None:
            continue
        digest = hashlib.sha256()
        counts = {name: 0 for name in class_names}
        for path, label in dataset.items:
            content = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    content.update(chunk)
            record = [path.relative_to(data_root).as_posix(), class_names[label], content.hexdigest()]
            digest.update((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
            counts[class_names[label]] += 1
        splits[split] = {
            "sha256": digest.hexdigest(),
            "count": len(dataset),
            "class_counts": counts,
        }
        combined.update(f"{split}:{digest.hexdigest()}\n".encode("utf-8"))
    return {
        "algorithm": "sha256-relative-path-label-content-v1",
        "sha256": combined.hexdigest(),
        "splits": splits,
    }


def seed_worker(_worker_id):
    """Make Python/NumPy augmentation RNGs reproducible in DataLoader workers."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def save_candidate(output: Path, checkpoint, metrics):
    """Persist each new best checkpoint immediately, using atomic replacements."""
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)

    metrics_path = output.with_suffix(".metrics.json")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{metrics_path.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary_path.replace(metrics_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/split"))
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/wallpaper.pt"),
        help="candidate path; each new validation best is saved immediately",
    )
    parser.add_argument(
        "--model",
        choices=["small-cnn", "resnet18", "resnet18-feature", "resnet50", "resnet50-feature"],
        default="resnet18",
    )
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained", dest="pretrained", action="store_true",
        help="use ImageNet weights for ResNet (default)",
    )
    pretrained_group.add_argument(
        "--no-pretrained", dest="pretrained", action="store_false",
        help="explicitly train without ImageNet weights (not valid for frozen features)",
    )
    parser.set_defaults(pretrained=True)
    parser.add_argument("--preprocessing", choices=PREPROCESSING_MODES, default="center_crop")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance", choices=["sampler", "loss", "none"], default="sampler",
        help="inverse-frequency sampler (default), weighted loss, or none; use one correction",
    )
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs)")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 8:
        parser.error("epochs/batch-size must be positive and image-size must be at least 8")
    if args.workers < 0 or args.threads < 0 or args.lr <= 0 or args.patience < 1:
        parser.error("workers/threads must be nonnegative; lr/patience must be positive")
    if not 0 <= args.label_smoothing < 1:
        parser.error("label-smoothing must be in [0, 1)")
    if args.model.endswith("-feature") and not args.pretrained:
        parser.error("frozen feature training requires pretrained weights")
    if args.model == "small-cnn":
        args.pretrained = False
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.threads:
        torch.set_num_threads(args.threads)
    selected_device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(selected_device)

    train_root = args.data / "train"
    class_names = sorted(directory.name for directory in train_root.iterdir() if directory.is_dir())
    if len(class_names) < 2:
        raise SystemExit(f"need at least 2 class folders under {train_root}")
    train_dataset = ImageFolderDataset(
        train_root, class_names, args.image_size, True, args.preprocessing
    )
    val_dataset = ImageFolderDataset(
        args.data / "val", class_names, args.image_size, preprocessing=args.preprocessing
    )
    test_dataset = (
        ImageFolderDataset(
            args.data / "test", class_names, args.image_size, preprocessing=args.preprocessing
        )
        if (args.data / "test").exists() else None
    )
    fingerprint = dataset_fingerprint(
        {"train": train_dataset, "val": val_dataset, "test": test_dataset}, args.data, class_names
    )
    for split in ("train", "val"):
        empty_classes = [
            name for name, count in fingerprint["splits"][split]["class_counts"].items() if count == 0
        ]
        if empty_classes:
            raise ValueError(f"{split} has no images for classes: {empty_classes}")
    print("data fingerprint:", fingerprint["sha256"], flush=True)

    labels = np.asarray([label for _, label in train_dataset.items], dtype=np.int64)
    counts = np.bincount(labels, minlength=len(class_names)).astype(np.int64)
    if np.any(counts == 0):
        missing = [name for index, name in enumerate(class_names) if counts[index] == 0]
        raise ValueError(f"train has no images for classes: {missing}")
    print("train class counts:", dict(zip(class_names, counts.tolist())), flush=True)
    sampler = None
    if args.balance == "sampler":
        weights = torch.as_tensor((1.0 / counts)[labels], dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    print("class balancing:", args.balance, flush=True)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset, sampler=sampler, shuffle=sampler is None,
        generator=torch.Generator().manual_seed(args.seed + 1), **loader_options,
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False,
        generator=torch.Generator().manual_seed(args.seed + 2), **loader_options,
    )
    test_loader = (
        DataLoader(
            test_dataset, shuffle=False,
            generator=torch.Generator().manual_seed(args.seed + 3), **loader_options,
        )
        if test_dataset is not None else None
    )

    # A requested architecture/weight source must either work or fail explicitly.
    model = build_model(args.model, len(class_names), args.pretrained).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    loss_weights = None
    if args.balance == "loss":
        class_weights = (counts.sum() / (len(class_names) * counts)).astype(np.float32)
        loss_weights = torch.as_tensor(class_weights, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=args.label_smoothing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.4, patience=2
    )

    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    config_json = json.dumps(config, ensure_ascii=False, sort_keys=True)
    best_f1 = -1.0
    history = []
    checkpoint = metrics = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen = correct = 0
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            correct += int((logits.argmax(1) == labels).sum())

        val_accuracy, val_matrix = evaluate(model, val_loader, device, len(class_names))
        val_f1, val_balanced_accuracy = classification_metrics(val_matrix)
        scheduler.step(val_f1)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "train_accuracy": correct / seen,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_f1,
            "val_balanced_accuracy": val_balanced_accuracy,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(row, flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            stale_epochs = 0
            validation_per_class = per_class_metrics(val_matrix, class_names)
            checkpoint = {
                "format_version": 2,
                "class_names": class_names,
                "image_size": args.image_size,
                "preprocessing": args.preprocessing,
                "model": args.model,
                "state_dict": {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                },
                "config": config,
                "config_json": config_json,
                "config_serialized": config_json,
                "device": str(device),
                "selection_metric": SELECTION_METRIC,
                "best_epoch": epoch,
                "best_val_accuracy": val_accuracy,
                "best_val_macro_f1": val_f1,
                "best_val_balanced_accuracy": val_balanced_accuracy,
                "val_confusion_matrix": val_matrix.tolist(),
                "val_per_class": validation_per_class,
                "data_fingerprint": fingerprint,
                "training_complete": False,
            }
            metrics = {
                key: value for key, value in checkpoint.items() if key != "state_dict"
            }
            metrics.update({
                "test_accuracy": None,
                "test_macro_f1": None,
                "test_balanced_accuracy": None,
                "confusion_matrix": None,
                "test_per_class": None,
                "history": list(history),
            })
            save_candidate(args.output, checkpoint, metrics)
            print(f"saved candidate {args.output} (epoch={epoch}, {SELECTION_METRIC}={val_f1:.4f})", flush=True)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping at epoch {epoch} (best macro-F1={best_f1:.4f})", flush=True)
                break

    if checkpoint is None or metrics is None:
        raise RuntimeError("training produced no candidate checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if test_loader is not None:
        test_accuracy, test_matrix = evaluate(model, test_loader, device, len(class_names))
        test_f1, test_balanced_accuracy = classification_metrics(test_matrix)
        metrics.update({
            "test_accuracy": test_accuracy,
            "test_macro_f1": test_f1,
            "test_balanced_accuracy": test_balanced_accuracy,
            "confusion_matrix": test_matrix.tolist(),
            "test_per_class": per_class_metrics(test_matrix, class_names),
        })
    checkpoint["training_complete"] = True
    metrics["training_complete"] = True
    metrics["history"] = history
    save_candidate(args.output, checkpoint, metrics)
    print(f"saved {args.output} (device={device}, best_epoch={checkpoint['best_epoch']})", flush=True)


if __name__ == "__main__":
    main()
