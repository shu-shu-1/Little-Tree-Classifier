"""Build a versioned training tree from review records without changing source files.

Known photographic series are kept out of earlier splits (test > val > train).
No personal external-test files are accepted as supplemental training images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=Path('data/split_capped'))
    ap.add_argument('--review', type=Path, default=Path('reports/label_review.jsonl'))
    ap.add_argument('--supplement', type=Path, help='Publicly collected class directories only')
    ap.add_argument('--supplement-review', type=Path, help='JSONL review; only action=keep is included')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--max-side', type=int, default=640)
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        ap.error('output must be empty; choose a new dataset version')
    if args.output.resolve() == args.input.resolve() or args.input.resolve() in args.output.resolve().parents:
        ap.error('output must be outside input')
    if args.max_side < 256:
        ap.error('max-side must be at least 256')
    reviews = [json.loads(x) for x in args.review.read_text(encoding='utf-8').splitlines() if x.strip()]
    review_map = {str(Path(r['path']).resolve()): r for r in reviews}
    priority = {'train': 0, 'val': 1, 'test': 2}
    groups = {}
    for r in reviews:
        group = r.get('visual_duplicate_group')
        if group:
            groups[group] = max(groups.get(group, 0), priority[r['split']])
    candidates, excluded = [], []
    for split in ['test', 'val', 'train']:
        for p in sorted((args.input / split).rglob('*')):
            if not p.is_file() or p.suffix.lower() not in EXTS:
                continue
            review = review_map.get(str(p.resolve()), {})
            reason = None
            if review.get('action') == 'exclude':
                reason = review['reason']
            group = review.get('visual_duplicate_group')
            if group and groups[group] > priority[split]:
                reason = 'visual series present in a held-out split: ' + group
            if reason:
                excluded.append({'path': str(p), 'reason': reason})
                continue
            label = review.get('label_reviewed') or p.parent.name
            candidates.append((p, split, label, review))
    supplement_review = {}
    if args.supplement_review:
        supplement_review = {str(Path(r['path']).resolve()): r for r in
                             (json.loads(x) for x in args.supplement_review.read_text(encoding='utf-8').splitlines() if x.strip())}
    if args.supplement:
        # Split by deterministic file digest; collection is independent of the
        # personal image archive. Repeating the builder gives the same splits.
        for folder in sorted(p for p in args.supplement.iterdir() if p.is_dir()):
            files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
            files.sort(key=lambda p: hashlib.sha256(p.read_bytes()).hexdigest())
            n = max(1, round(len(files) * .15)) if len(files) >= 5 else 0
            for i, p in enumerate(files):
                review = supplement_review.get(str(p.resolve()), {})
                if review and review.get('action') != 'keep':
                    excluded.append({'path': str(p), 'reason': review.get('reason', 'supplement review excluded')})
                    continue
                split = 'test' if i < n else ('val' if i < 2 * n else 'train')
                candidates.append((p, split, folder.name, review or {'source': 'public_supplement'}))
    candidates.sort(key=lambda row: -priority[row[1]])
    pixel_seen, records, counts = set(), [], Counter()
    for source, split, label, review in candidates:
        try:
            with Image.open(source) as im:
                im.draft('RGB', (args.max_side, args.max_side))
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
                digest = hashlib.sha256(im.tobytes() + str(im.size).encode()).hexdigest()
                if digest in pixel_seen:
                    excluded.append({'path': str(source), 'reason': 'identical normalized pixels'})
                    continue
                pixel_seen.add(digest)
                target = args.output / split / label / (digest[:24] + '.jpg')
                target.parent.mkdir(parents=True, exist_ok=True)
                im.save(target, 'JPEG', quality=94)
            records.append({'file': target.relative_to(args.output).as_posix(), 'source': str(source.resolve()),
                            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'split': split,
                            'label': label, 'review': review, 'pixel_sha256': digest})
            counts[split + '/' + label] += 1
        except (OSError, ValueError) as exc:
            excluded.append({'path': str(source), 'reason': str(exc)})
    if not records:
        ap.error('no images were written')
    (args.output / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    summary = {'counts': dict(sorted(counts.items())), 'excluded': excluded,
               'review_sha256': hashlib.sha256(args.review.read_bytes()).hexdigest(),
               'known_series_policy': 'test > val > train; earlier split copies excluded',
               'limitations': 'Only visually reviewed series and identical normalized pixels removed; other near duplicates may remain.'}
    (args.output / 'build_report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'counts': summary['counts'], 'excluded': len(excluded)}, indent=2))


if __name__ == '__main__':
    main()
