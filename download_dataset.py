#!/usr/bin/env python3
"""Download openly licensed wallpaper images from Openverse.

The downloader intentionally uses a public API instead of scraping search
engines.  It stores a ``metadata.jsonl`` file next to the images so that the
license and attribution information remains available after training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import requests
import re
from PIL import Image, ImageFile
from io import BytesIO
from urllib.parse import urljoin
from bs4 import BeautifulSoup

ImageFile.LOAD_TRUNCATED_IMAGES = True
LOG = logging.getLogger("download_dataset")
API = "https://api.openverse.org/v1/images/"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
NETBIAN_BASE = "https://pic.netbian.com/"
SIMPLE_BASE = "https://simpledesktops.com/"
WALLHAVEN_API = "https://wallhaven.cc/api/v1/search"
NASA_API = "https://images-api.nasa.gov/search"
DEFAULT_CATEGORIES = {
    "anime": "anime illustration",
    "landscape": "landscape wallpaper",
    # Commons search performs an AND over terms; keep this broad enough to
    # return results, while metadata marks the query as a weak label for review.
    "pets": "cat",
    "nature": "nature flowers wallpaper",
    "city": "city architecture wallpaper",
    "space": "space astronomy wallpaper",
    "cars": "car automobile wallpaper",
    "abstract": "abstract wallpaper",
}


def request_json(session: requests.Session, url: str, params: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries - 1:
                raise
            LOG.warning("request failed (%s), retrying", exc)
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def iter_results(session: requests.Session, query: str, pages: int, page_size: int) -> Iterable[dict]:
    for page in range(1, pages + 1):
        payload = request_json(session, API, {"q": query, "page": page, "page_size": page_size})
        results = payload.get("results", [])
        if not results:
            break
        yield from results


def iter_wallhaven(session: requests.Session, query: str, pages: int, page_size: int) -> Iterable[dict]:
    """Yield Wallhaven search results (public API; API key is optional for basic search)."""
    # Wallhaven caps page size at 24 and uses a 1-based page parameter.
    for page in range(1, pages + 1):
        payload = request_json(session, WALLHAVEN_API,
                               {"q": query, "page": page, "sorting": "relevance"})
        results = payload.get("data", [])
        if not results:
            break
        for item in results[:page_size]:
            url = item.get("path") or item.get("url")
            if not url:
                continue
            yield {
                "thumbnail": (item.get("thumbs") or {}).get("large") or url,
                "url": url,
                "foreign_landing_url": item.get("url"),
                "license": "wallhaven terms (check source before redistribution)",
                "license_url": "https://wallhaven.cc/terms",
                "creator": item.get("username"),
                "_source": "wallhaven",
            }


def iter_nasa(session: requests.Session, query: str, pages: int, page_size: int) -> Iterable[dict]:
    """Yield NASA image-library results (public-domain metadata where indicated)."""
    for page in range(1, pages + 1):
        payload = request_json(session, NASA_API,
                               {"q": query, "media_type": "image", "page": page,
                                "page_size": min(page_size, 100)})
        collection = payload.get("collection", {})
        items = collection.get("items", [])
        if not items:
            break
        for item in items:
            data = (item.get("data") or [{}])[0]
            links = item.get("links") or []
            image_url = next((x.get("href") for x in links if x.get("href")), None)
            if not image_url:
                continue
            yield {
                "thumbnail": image_url,
                "url": image_url,
                "foreign_landing_url": item.get("href"),
                "license": "NASA public domain (verify restrictions)",
                "license_url": "https://www.nasa.gov/multimedia/guidelines/index.html",
                "creator": data.get("photographer") or data.get("secondary_creator"),
                "_source": "nasa",
            }


def iter_html_source(session: requests.Session, query: str, pages: int,
                     page_size: int, site: str) -> Iterable[dict]:
    """Best-effort extraction from public wallpaper search pages.

    These sites do not expose stable APIs. We only read ordinary HTML pages,
    honor the caller's delay, and retain the landing URL for attribution.
    """
    templates = {
        "wallpaperflare": "https://wallpaperflare.com/search?wallpaper={q}&page={page}",
        "wallpapercave": "https://wallpapercave.com/search?q={q}&page={page}",
        # WallpaperBetter serves a Chinese localized wallpaper catalogue.
        "wallpaperbetter": "https://wallpaperbetter.com/search?q={q}&page={page}",
        "netbian": "https://pic.netbian.com/e/search/result/?searchid=1&keyboard={q}&page={page}",
    }
    for page in range(1, pages + 1):
        url = templates[site].format(q=requests.utils.quote(query), page=page)
        try:
            response = session.get(url, timeout=30, headers={"Accept": "text/html"})
            response.raise_for_status()
        except requests.RequestException as exc:
            LOG.warning("%s search unavailable: %s", site, exc)
            return
        # Capture src/data-src/lazy URLs. Restrict to common image extensions and
        # skip tiny tracking assets. HTML entities are decoded by the URL parser.
        matches = re.findall(r'(?:src|data-src|data-original)=["\']([^"\']+)', response.text,
                            flags=re.IGNORECASE)
        emitted = 0
        for image_url in matches:
            image_url = image_url.replace("&amp;", "&")
            if image_url.startswith("//"):
                image_url = "https:" + image_url
            if not image_url.startswith(("http://", "https://")):
                continue
            if not re.search(r'\.(?:jpe?g|png|webp)(?:[?#]|$)', image_url, re.I):
                continue
            yield {
                "thumbnail": image_url,
                "url": image_url,
                "foreign_landing_url": url,
                "license": "unknown (verify site terms)",
                "license_url": url,
                "creator": None,
                "_source": site,
            }
            emitted += 1
            if emitted >= page_size:
                break


def iter_commons(session: requests.Session, query: str, pages: int, page_size: int) -> Iterable[dict]:
    """Yield Wikimedia Commons bitmap search results in Openverse-like shape."""
    cont: dict = {}
    for _ in range(pages):
        params = {
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6,
            "gsrlimit": min(page_size, 50), "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata", "iiurlwidth": 768,
            **cont,
        }
        payload = request_json(session, COMMONS_API, params)
        pages_data = payload.get("query", {}).get("pages", {})
        for page in pages_data.values():
            info = (page.get("imageinfo") or [{}])[0]
            if not info.get("url") and not info.get("thumburl"):
                continue
            ext = info.get("extmetadata") or {}
            def ext_value(name: str):
                value = ext.get(name, {})
                return value.get("value") if isinstance(value, dict) else value
            yield {
                "thumbnail": info.get("thumburl") or info.get("url"),
                "url": info.get("url"),
                "foreign_landing_url": "https://commons.wikimedia.org/wiki?curid=" + str(page.get("pageid", "")),
                "license": ext_value("LicenseShortName"),
                "license_url": ext_value("LicenseUrl"),
                "creator": ext_value("Artist"),
                "_source": "wikimedia_commons",
            }
        cont = payload.get("continue", {})
        if not cont:
            break


def iter_netbian(session: requests.Session, category: str, pages: int = 5) -> Iterable[dict]:
    """Parse public thumbnail listings from 彼岸图网 (pic.netbian.com).

    The site exposes category pages without login; only listing images are
    consumed and requests are rate limited by the caller.  ``category`` is
    mapped to the site's Chinese URL slugs where possible.
    """
    # ``category`` is usually the textual query (e.g. "anime illustration")
    # when called through the generic source iterator.
    q = category.lower()
    slug = ("4kdongman" if any(k in q for k in ("anime", "illustration", "动漫")) else
            "4kdongwu" if any(k in q for k in ("pet", "cat", "dog", "动物")) else
            "4kqiche" if any(k in q for k in ("car", "auto", "汽车")) else "4kfengjing")
    for page in range(1, pages + 1):
        path = f"{slug}/" if page == 1 else f"{slug}/index_{page}.html"
        try:
            response = session.get(urljoin(NETBIAN_BASE, path), timeout=30)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "gbk"
            soup = BeautifulSoup(response.text, "html.parser")
        except requests.RequestException:
            continue
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if not src or "/uploads/" not in src:
                continue
            yield {"url": urljoin(NETBIAN_BASE, src), "thumbnail": urljoin(NETBIAN_BASE, src),
                   "foreign_landing_url": urljoin(NETBIAN_BASE, path), "license": "unknown",
                   "creator": "", "_source": "pic_netbian"}


def iter_simpledesktops(session: requests.Session, pages: int = 5) -> Iterable[dict]:
    """Parse Simple Desktops public browse pages and derive original assets."""
    for page in range(1, pages + 1):
        path = "browse/" if page == 1 else f"browse/{page}/"
        try:
            response = session.get(urljoin(SIMPLE_BASE, path), timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
        except requests.RequestException:
            continue
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if not src or "static.simpledesktops.com/uploads/" not in src:
                continue
            # Thumbnails are named ``<original>.295x184_q100.png``.
            original = src.replace(".295x184_q100.png", "")
            yield {"url": original, "thumbnail": original,
                   "foreign_landing_url": urljoin(SIMPLE_BASE, path),
                   "license": "unknown", "creator": "", "_source": "simpledesktops"}


def decode_image(content: bytes) -> tuple[bytes, int, int] | None:
    try:
        with Image.open(BytesIO(content)) as image:
            image = image.convert("RGB")
            width, height = image.size
            if min(width, height) < 256 or max(width, height) / min(width, height) > 4.0:
                return None
            output = BytesIO()
            image.save(output, format="JPEG", quality=92, optimize=True)
            return output.getvalue(), width, height
    except Exception:
        return None


def _source_iterators(session: requests.Session, query: str, pages: int, page_size: int,
                      source: str) -> list[tuple[str, Iterable[dict]]]:
    """Return source iterators in requested order. ``auto`` uses all sources."""
    names = [s.strip().lower() for s in source.split(",") if s.strip()]
    if not names or "auto" in names:
        # Safe default: sources with explicit open licensing metadata.
        names = ["openverse", "commons", "nasa"]
    out: list[tuple[str, Iterable[dict]]] = []
    for name in names:
        try:
            if name == "openverse":
                out.append((name, iter_results(session, query, pages, page_size)))
            elif name in {"commons", "wikimedia", "wikimedia_commons"}:
                out.append(("wikimedia_commons", iter_commons(session, query, pages, page_size)))
            elif name == "nasa":
                out.append((name, iter_nasa(session, query, pages, page_size)))
            elif name in {"wallhaven-unsafe"}:
                out.append((name, iter_wallhaven(session, query, pages, page_size)))
            elif name in {"wallpaperflare-unsafe", "wallpapercave-unsafe", "wallpaperbetter-unsafe"}:
                site = name.replace("-unsafe", "")
                out.append((name, iter_html_source(session, query, pages, page_size, site)))
            elif name in {"netbian-unsafe"}:
                # Public category listings on pic.netbian.com; terms/licensing
                # are not machine-readable, so metadata marks them unknown.
                out.append((name, iter_netbian(session, query, pages)))
            elif name in {"simpledesktops-unsafe"}:
                out.append((name, iter_simpledesktops(session, pages)))
            else:
                LOG.warning("unknown source '%s'; skipping", name)
        except Exception as exc:
            LOG.warning("failed to initialize source %s: %s", name, exc)
    return out


def download_category(session: requests.Session, root: Path, category: str, query: str,
                      target: int, pages: int, page_size: int, delay: float,
                      seen: set[str], metadata_file, source: str = "auto") -> int:
    folder = root / "raw" / category
    folder.mkdir(parents=True, exist_ok=True)
    count = 0
    existing = len(list(folder.glob("*.jpg")))
    if existing >= target:
        LOG.info("%s already has %d/%d images; skipping", category, existing, target)
        return 0
    remaining = target - existing
    try:
        for source_name, results in _source_iterators(session, query, pages, page_size, source):
            if count >= remaining:
                break
            LOG.info("%s: querying %s", category, source_name)
            try:
                for item in results:
                    if count >= remaining:
                        break
                    url = item.get("thumbnail") or item.get("url")
                    if not url or not str(url).startswith(("http://", "https://")):
                        continue
                    try:
                        response = session.get(url, timeout=30, headers={"Accept": "image/*"})
                        response.raise_for_status()
                    except requests.RequestException:
                        continue
                    decoded = decode_image(response.content)
                    if decoded is None:
                        continue
                    image_bytes, width, height = decoded
                    digest = hashlib.sha256(image_bytes).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    path = folder / f"{digest[:16]}.jpg"
                    path.write_bytes(image_bytes)
                    record = {
                        "file": str(path.relative_to(root)).replace("\\", "/"),
                        "category": category,
                        "sha256": digest,
                        "width": width,
                        "height": height,
                        "source_url": item.get("url"),
                        "page_url": item.get("foreign_landing_url") or item.get("source"),
                        "license": item.get("license"),
                        "license_url": item.get("license_url"),
                        "creator": item.get("creator"),
                        "query": query,
                        "source": item.get("_source", source_name),
                        "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    }
                    metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    metadata_file.flush()
                    count += 1
                    if count % 25 == 0:
                        LOG.info("%s: %d/%d", category, count, target)
                    time.sleep(delay)
            except requests.RequestException as exc:
                LOG.warning("%s source failed for %s: %s", source_name, category, exc)
    except Exception as exc:
        LOG.warning("source iteration failed for %s: %s", category, exc)
    if count < remaining:
        LOG.warning("%s: downloaded %d new images; target shortfall %d", category, count, remaining - count)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--per-class", type=int, default=200, help="target images per class")
    parser.add_argument("--pages", type=int, default=10)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between downloads")
    parser.add_argument("--categories", nargs="*", help="category names; defaults to all built-ins")
    parser.add_argument("--user-agent", default="wallpaper-classifier/1.0 (research; contact local)")
    parser.add_argument("--source", default="auto",
                        help="comma-separated sources: auto (openverse,commons,nasa), or openverse, commons, nasa, wallhaven-unsafe, wallpaperflare-unsafe, wallpaperbetter-unsafe, netbian-unsafe, simpledesktops-unsafe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested = args.categories or list(DEFAULT_CATEGORIES)
    unknown = set(requested) - set(DEFAULT_CATEGORIES)
    if unknown:
        raise SystemExit(f"unknown categories: {', '.join(sorted(unknown))}")
    categories = {name: DEFAULT_CATEGORIES[name] for name in requested}
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "raw" / "metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    # Existing files are retained, making interrupted downloads resumable.
    for path in (args.output / "raw").glob("*/*.jpg"):
        try:
            seen.add(hashlib.sha256(path.read_bytes()).hexdigest())
        except OSError:
            pass
    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent, "Accept": "application/json"})
    with metadata_path.open("a", encoding="utf-8") as metadata_file:
        for category, query in categories.items():
            got = download_category(session, args.output, category, query, args.per_class,
                                    args.pages, args.page_size, args.delay, seen, metadata_file,
                                    args.source)
            LOG.info("finished %s: %d images", category, got)
    totals = {c: len(list((args.output / "raw" / c).glob("*.jpg"))) for c in categories}
    LOG.info("dataset totals: %s", totals)
    return 0 if all(v > 0 for v in totals.values()) else 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
