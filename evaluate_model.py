"""Evaluate checkpoints on a fixed folder split or visually reviewed JSONL manifest.

Personal photos remain read-only. Reviewed external labels are never inferred
from directory names or a model prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import load_checkpoint, preprocess_image

EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.avif'}


def metrics_for(matrix):
    cm = np.asarray(matrix, dtype=np.int64)
    tp = cm.diagonal()
    support, predicted = cm.sum(1), cm.sum(0)
    recall = np.divide(tp, support, out=np.zeros(len(tp)), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(len(tp)), where=predicted > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(len(tp)), where=(support + predicted) > 0)
    valid = support > 0
    return {'count': int(cm.sum()), 'accuracy': float(tp.sum() / cm.sum()) if cm.sum() else None,
            'macro_f1': float(f1[valid].mean()) if valid.any() else None,
            'balanced_accuracy': float(recall[valid].mean()) if valid.any() else None,
            'confusion_matrix': cm.tolist(), 'recall': recall.tolist(),
            'precision': precision.tolist(), 'support': support.tolist()}


def collect(args, classes):
    if args.manifest:
        rows = [json.loads(line) for line in args.manifest.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        return [r for r in rows if r.get('label') in classes and r.get('action') != 'exclude'
                and r.get('eligible_for_closed_set', True) and not r.get('exact_training_overlap', False)]
    return [{'path': str(p.resolve()), 'label': c} for c in classes
            for p in sorted((args.data / c).rglob('*')) if p.is_file() and p.suffix.lower() in EXTS]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--data', type=Path, help='Fixed test folder with class subfolders')
    source.add_argument('--manifest', type=Path, help='JSONL with path, label, optional accepted_labels')
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--exclude-duplicates-of', type=Path, help='Training tree for exact byte overlap checks')
    args = ap.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        ap.error('threads and batch size must be positive')
    torch.set_num_threads(args.threads)
    model, ck = load_checkpoint(args.checkpoint, 'cpu')
    classes = ck['class_names']
    rows = collect(args, classes)
    if not rows:
        ap.error('no labeled images found')
    excluded_hashes = set()
    if args.exclude_duplicates_of:
        for p in args.exclude_duplicates_of.rglob('*'):
            if p.is_file() and p.suffix.lower() in EXTS:
                excluded_hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    results, skipped, inputs, pending = [], [], [], []
    seen = set()

    def flush():
        if not inputs:
            return
        with torch.inference_mode():
            probabilities = model(torch.stack(inputs)).softmax(1).cpu().tolist()
        for row, probs in zip(pending, probabilities):
            idx = int(np.argmax(probs))
            actual = classes.index(row['label'])
            matrix[actual, idx] += 1
            results.append({**row, 'prediction': classes[idx], 'confidence': probs[idx],
                            'correct': classes[idx] == row['label'],
                            'accepted_correct': classes[idx] in row.get('accepted_labels', [row['label']]),
                            'probabilities': dict(zip(classes, probs))})
        inputs.clear()
        pending.clear()

    for row in rows:
        path = Path(row['path'])
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in excluded_hashes or digest in seen:
                skipped.append({'path': str(path), 'reason': 'exact_duplicate'})
                continue
            if row.get('sha256') and row['sha256'] != digest:
                raise ValueError('image changed since review')
            seen.add(digest)
            with Image.open(path) as im:
                x = preprocess_image(im, ck['image_size'], preprocessing=ck.get('preprocessing', 'stretch'))
            inputs.append(x)
            pending.append({**row, 'sha256': digest})
        except Exception as exc:
            skipped.append({'path': str(path), 'reason': str(exc)})
        if len(inputs) >= args.batch_size:
            flush()
    flush()
    summary = metrics_for(matrix)
    summary.update({'checkpoint': str(args.checkpoint.resolve()),
                    'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    'preprocessing': ck.get('preprocessing', 'stretch'), 'class_names': classes,
                    'accepted_label_accuracy': sum(r['accepted_correct'] for r in results) / len(results) if results else None,
                    'label_counts': dict(Counter(r['label'] for r in results)),
                    'skipped_count': len(skipped)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'summary': summary, 'predictions': results, 'skipped': skipped}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not results:
        raise SystemExit('all inputs were skipped; inspect evaluation report')


if __name__ == '__main__':
    main()
