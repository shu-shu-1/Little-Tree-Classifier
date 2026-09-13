"""Read-only perceptual-hash search for review-set training contamination."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageOps, ImageDraw

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
rows = [json.loads(line) for line in (OUT / "external_reviewed.jsonl").read_text(encoding="utf-8").splitlines()]
# 64-bit DCT perceptual hash; geometric/color variants still need visual review.
k = np.arange(8)[:, None]
n = np.arange(32)[None, :]
matrix = np.cos(np.pi * (2*n + 1)*k / 64)
def phash(path):
    with Image.open(path) as im:
        a = np.asarray(ImageOps.exif_transpose(im).convert("L").resize((32,32), Image.Resampling.LANCZOS), dtype=np.float64)
    dct = matrix @ a @ matrix.T
    bits = dct > np.median(dct.flatten()[1:])
    return int.from_bytes(np.packbits(bits).tobytes(), "big")

external = [(row, phash(row["path"])) for row in rows if row["eligible_for_closed_set"]]
matches = []
count = 0
for folder in [ROOT / "data" / "raw", ROOT / "data" / "split"]:
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        try:
            digest = phash(path)
        except Exception:
            continue
        count += 1
        for row, query in external:
            distance = (digest ^ query).bit_count()
            if distance <= 6:
                matches.append({"external_id": row["id"], "external_path": row["path"], "training_path": str(path), "phash_hamming": distance, "status": "requires_visual_review"})

matches.sort(key=lambda row: (row["external_id"], row["phash_hamming"]))
(OUT / "external_nearduplicate_candidates.json").write_text(json.dumps({"training_images_scanned": count, "matches": matches}, ensure_ascii=False, indent=2), encoding="utf-8")
for start in range(0, len(matches), 8):
    subset = matches[start:start+8]
    sheet = Image.new("RGB", (840, len(subset)*180), "#eee")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(subset):
        for side, key in enumerate(["external_path", "training_path"]):
            with Image.open(row[key]) as im:
                thumb = ImageOps.contain(ImageOps.exif_transpose(im).convert("RGB"), (405, 153))
            sheet.paste(thumb, (side*420+(420-thumb.width)//2, i*180))
        draw.text((8, i*180+158), f"Pair {start+i}, external ID {row['external_id']}, distance {row['phash_hamming']}", fill="black")
        draw.text((425, i*180+158), "training: " + Path(row["training_path"]).parent.name, fill="black")
    sheet.save(OUT / f"external_nearduplicate_pairs_{start//8+1:02d}.jpg", quality=92)
print(json.dumps({"training_images_scanned": count, "candidate_pairs": len(matches), "external_ids": sorted({row["external_id"] for row in matches})}))
