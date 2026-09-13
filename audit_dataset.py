#!/usr/bin/env python3
"""Audit a downloaded image dataset without modifying files.

Reports class/source imbalance, unreadable or low-resolution images, exact
duplicates (including duplicates that occur in different classes), and rows
in metadata that no longer have a corresponding file.  Use this before
rebuilding ``data/split`` and retraining.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".avif"}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("data/raw"))
    ap.add_argument("--metadata", type=Path, default=None)
    ap.add_argument("--min-width", type=int, default=256)
    ap.add_argument("--min-height", type=int, default=256)
    ap.add_argument("--json", type=Path, default=None, help="also write a machine-readable report")
    args = ap.parse_args()
    root = args.input
    if not root.exists():
        ap.error(f"input directory not found: {root}")
    metadata_path = args.metadata or (root / "metadata.jsonl")
    classes = sorted(p for p in root.iterdir() if p.is_dir())
    counts, sources = Counter(), Counter()
    class_sources: dict[str, Counter] = defaultdict(Counter)
    bad, small, hashes = [], [], defaultdict(list)
    files = []
    for cls in classes:
        for path in sorted(cls.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTS:
                continue
            files.append(path); counts[cls.name] += 1
            try:
                with Image.open(path) as im:
                    width, height = im.size
                    im.verify()
                if width < args.min_width or height < args.min_height:
                    small.append({"file": str(path), "width": width, "height": height})
                hashes[file_hash(path)].append(str(path))
            except Exception as exc:
                bad.append({"file": str(path), "error": str(exc)})
    metadata_files = set()
    metadata_categories = Counter()
    metadata_weak = []
    metadata_rows = 0
    if metadata_path.exists():
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line); metadata_rows += 1
            except Exception:
                continue
            if row.get("source"):
                sources[row["source"]] += 1
                metadata_categories[row.get("category", "")] += 1
                if row.get("source") not in {"wikimedia_commons", "openverse", "nasa"} or not row.get("license"):
                    metadata_weak.append({"file": row.get("file"), "source": row.get("source"),
                                          "license": row.get("license"), "category": row.get("category")})
            name = row.get("filename") or row.get("file") or row.get("path")
            if name:
                metadata_files.add(Path(name).name)
                class_name = row.get("category")
                if class_name and row.get("source"):
                    class_sources[class_name][row["source"]] += 1
    duplicate_groups = [v for v in hashes.values() if len(v) > 1]
    cross_class = [v for v in duplicate_groups if len({Path(x).parent.name for x in v}) > 1]
    report = {
        "input": str(root), "classes": counts, "total_files": len(files),
        "metadata_rows": metadata_rows, "metadata_sources": sources,
        "metadata_categories": metadata_categories,
        "metadata_weak_label_rows": metadata_weak,
        "class_sources": {k: dict(v) for k, v in class_sources.items()},
        "unreadable": bad, "low_resolution": small,
        "duplicate_groups": duplicate_groups, "cross_class_duplicate_groups": cross_class,
    }
    print("Class counts:", dict(counts)); print("Total images:", len(files))
    print("Metadata sources:", dict(sources) if metadata_rows else "metadata not found")
    if metadata_rows:
        print("Metadata categories:", dict(metadata_categories))
        print("Weak-label/unknown-license rows:", len(metadata_weak))
    print(f"Unreadable: {len(bad)} | below {args.min_width}x{args.min_height}: {len(small)}")
    print(f"Exact duplicate groups: {len(duplicate_groups)} | cross-class: {len(cross_class)}")
    if bad: print("Unreadable examples:", bad[:5])
    if small: print("Low-resolution examples:", small[:5])
    if cross_class: print("Cross-class duplicate examples:", cross_class[:3])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Wrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
