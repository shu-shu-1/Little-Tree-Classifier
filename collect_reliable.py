#!/usr/bin/env python3
"""Collect supplemental, openly licensed images from Wikimedia Commons.

This script is intentionally independent from ``download_dataset.py``.  It
uses several precise queries per minority class, stores the Commons title and
license metadata for every image, and downloads only bounded-size thumbnails.
The resulting manifest is suitable for a later human review before training.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False
LOG = logging.getLogger("collect_reliable")
API = "https://commons.wikimedia.org/w/api.php"

# Queries are deliberately specific to reduce weak labels.  Pets focuses on
# dogs and small companion animals because the existing set is cat-heavy.
QUERIES = {
    "city": [
        "city skyline", "urban skyline", "city architecture", "downtown buildings",
        "urban night photography", "historic city center", "city street architecture",
    ],
    "nature": [
        "flower botanical", "wildflower meadow", "forest landscape", "mountain landscape",
        "wildlife nature", "waterfall landscape", "lake landscape", "butterfly flower",
    ],
    "pets": [
        "dog portrait", "puppy dog", "golden retriever dog", "labrador dog",
        "small dog pet", "rabbit pet", "guinea pig pet", "hamster pet",
    ],
}


def commons_items(session: requests.Session, query: str, pages: int = 5,
                  page_size: int = 50) -> Iterable[dict]:
    continuation: dict = {}
    for _ in range(pages):
        params = {
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6,
            "gsrlimit": min(50, page_size), "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata", "iiurlwidth": 640,
            **continuation,
        }
        response = session.get(API, params=params, timeout=(10, 30))
        response.raise_for_status()
        payload = response.json()
        for page in (payload.get("query", {}).get("pages", {}) or {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            ext = info.get("extmetadata") or {}

            def value(key: str):
                obj = ext.get(key, {})
                return obj.get("value") if isinstance(obj, dict) else obj

            url = info.get("thumburl") or info.get("url")
            if not url:
                continue
            yield {
                "title": page.get("title", ""),
                "url": info.get("url") or url,
                "thumbnail_url": url,
                "landing_url": "https://commons.wikimedia.org/wiki?curid=" + str(page.get("pageid", "")),
                "license": value("LicenseShortName"),
                "license_url": value("LicenseUrl"),
                "creator": value("Artist"),
                "width": info.get("width"),
                "height": info.get("height"),
                "mime": info.get("mime"),
            }
        continuation = payload.get("continue", {}) or {}
        if not continuation:
            break


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return text[:90] or "image"


def fetch_image(session: requests.Session, url: str, max_bytes: int) -> tuple[bytes, Image.Image]:
    response = session.get(url, timeout=(10, 30), stream=True)
    response.raise_for_status()
    data = bytearray()
    for chunk in response.iter_content(64 * 1024):
        if chunk:
            data.extend(chunk)
            if len(data) > max_bytes:
                raise ValueError(f"image exceeds {max_bytes} bytes")
    image = Image.open(io.BytesIO(data))
    image.load()
    return bytes(data), image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=60, help="accepted images per class")
    parser.add_argument("--out", type=Path, default=Path("data/supplemental"))
    parser.add_argument("--pages", type=int, default=12)
    parser.add_argument("--delay", type=float, default=0.6)
    parser.add_argument("--max-bytes", type=int, default=5_000_000)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--classes", nargs="+", choices=sorted(QUERIES), default=sorted(QUERIES))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    session = requests.Session()
    session.headers.update({"User-Agent": "WallpaperClassifierResearch/1.0 (educational; Wikimedia Commons API)"})
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"
    # Start a fresh manifest for a reproducible collection, while preserving
    # any existing files under a different user-selected output directory.
    if manifest_path.exists():
        manifest_path.unlink()
    seen_hashes: set[str] = set()
    totals = {}
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for category in args.classes:
            folder = args.out / category
            folder.mkdir(parents=True, exist_ok=True)
            accepted = 0
            attempted = 0
            for query in QUERIES[category]:
                if accepted >= args.target:
                    break
                try:
                    candidates = commons_items(session, query, pages=args.pages)
                    for item in candidates:
                        if accepted >= args.target:
                            break
                        attempted += 1
                        # Commons metadata can contain non-images despite the
                        # bitmap query; reject those before writing a record.
                        if item.get("mime") and not str(item["mime"]).startswith("image/"):
                            continue
                        try:
                            raw, image = fetch_image(session, item["thumbnail_url"], args.max_bytes)
                            if image.width < 256 or image.height < 256:
                                continue
                            digest = hashlib.sha256(raw).hexdigest()
                            if digest in seen_hashes:
                                continue
                            seen_hashes.add(digest)
                            # Normalize to JPEG thumbnail, bounded to 768 px.
                            image = image.convert("RGB")
                            image.thumbnail((768, 768), Image.Resampling.LANCZOS)
                            filename = f"{accepted:04d}_{digest[:12]}_{safe_name(item['title'].removeprefix('File:'))}.jpg"
                            destination = folder / filename
                            image.save(destination, format="JPEG", quality=90, optimize=True)
                            record = {
                                "path": destination.as_posix(), "category": category,
                                "query": query, "title": item["title"],
                                "url": item["url"], "thumbnail_url": item["thumbnail_url"],
                                "landing_url": item["landing_url"],
                                "license": item["license"], "license_url": item["license_url"],
                                "creator": item["creator"], "sha256": digest,
                                "source": "wikimedia_commons", "review": "pending_manual_contact_sheet",
                            }
                            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                            manifest.flush()
                            accepted += 1
                            if accepted % 10 == 0:
                                LOG.info("%s: %d/%d", category, accepted, args.target)
                        except (requests.RequestException, ValueError, OSError) as exc:
                            LOG.debug("skip %s: %s", item.get("title"), exc)
                        time.sleep(max(0.0, args.delay))
                except (requests.RequestException, ValueError) as exc:
                    LOG.warning("query failed (%s / %s): %s", category, query, exc)
            totals[category] = {"accepted": accepted, "attempted": attempted}
            LOG.info("%s complete: %d accepted from %d candidates", category, accepted, attempted)
    (args.out / "collection_summary.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
