"""Read-only sampling of personal photos for a manually reviewed external test."""
from pathlib import Path
import json
import random
import sys
from PIL import Image, ImageOps, ImageDraw

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"D:\Users\张秫\onedrive\OneDrive - 小树科技\图片")
OUT = Path(__file__).resolve().parent
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
rng = random.Random(20260913)
selected = []
def add(paths, group):
    for p in paths:
        if p not in [x[0] for x in selected]:
            selected.append((p, group))
def pictures(path):
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in EXT)

for name in ["西安游·图", "2024.5.1-5.3西安", "本机照片"]:
    add(pictures(ROOT / name), name)
add(sorted(ROOT.glob("Bing_*.jpg")), "root_bing")
add(sorted(ROOT.glob("wallhaven-*.jpg")), "root_wallpaper")
add([ROOT / "spotlight21.jpg"], "root_wallpaper")
add(pictures(ROOT / "壁纸"), "wallpaper_folder")
root_candidates = [p for p in ROOT.iterdir() if p.is_file() and p.suffix.lower() in EXT]
add(rng.sample(sorted(root_candidates), min(24, len(root_candidates))), "root_random")
characters = ["八重神子", "白圣女与黑牧师", "冰之女皇", "芙宁娜︱芙芙", "灵砂", "麻衣学姐", "那维莱特 - 来自能猫", "其他原神图片", "千织", "阮·梅", "散兵 - 来自能猫", "丝柯克", "穗︱满穗", "温迪 - 来自能猫", "昔涟", "遐蝶", "叶瞬光︱小光师姐"]
character_candidates = sorted(p for name in characters for p in pictures(ROOT / name))
add(rng.sample(character_candidates, 28), "character_random")
manifest = []
for i, (path, group) in enumerate(selected):
    manifest.append({"id": i, "path": str(path), "sampling_group": group})
(OUT / "external_review_candidates.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
W, H, COLS, PER = 240, 205, 5, 25
for start in range(0, len(manifest), PER):
    subset = manifest[start:start+PER]
    sheet = Image.new("RGB", (COLS*W, ((len(subset)+COLS-1)//COLS)*H), "#e8e8e8")
    draw = ImageDraw.Draw(sheet)
    for local, item in enumerate(subset):
        x, y = (local % COLS)*W, (local//COLS)*H
        try:
            with Image.open(item["path"]) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                thumb = ImageOps.contain(image, (W-8, H-26))
                sheet.paste(thumb, (x+(W-thumb.width)//2, y+(H-26-thumb.height)//2))
                draw.text((x+7, y+H-21), f"ID {item['id']:03d}   {image.width}x{image.height}", fill="black")
        except Exception as exc:
            draw.text((x+7, y+H-21), f"ID {item['id']:03d} ERROR", fill="red")
            item["read_error"] = str(exc)
    target = OUT / f"external_review_sheet_{start//PER+1:02d}.jpg"
    sheet.save(target, quality=94)
    print(target)
print(f"selected={len(manifest)}")
