"""Shared cover loading + normalization.

Every feature extractor consumes the output of `load_cover` so that source resolution
is never a confound (see build-log.md 2026-09-04): strip goodreads' `_SX/_SY` downscale
suffix on download, then resize every cover to one canonical height.
"""
from __future__ import annotations

import hashlib
import io
import re
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# A handful of legitimate Goodreads covers are large scans (~226M px) that trip PIL's default
# decompression-bomb guard (178.9M px). Raise the ceiling rather than disable the check --
# every cover gets downscaled to CANON_H immediately after decode regardless of source size,
# so this only affects which files decode, not memory use downstream.
Image.MAX_IMAGE_PIXELS = 300_000_000

CANON_H = 480  # corpus resolution ceiling; every cover is resized to this height
RAW_CACHE = Path("data/covers_raw")
# goodreads bakes a downscale suffix into ~20% of URLs, in three forms seen in the data:
#   ._SY475_.jpg   ._SX318_.jpg   ._SX318_SY475_.jpg
# the suffix is `._` + one-or-more `S[XY]<n>_` groups, immediately before the extension.
_SUFFIX_RE = re.compile(r"\._(?:S[XY]\d+_)+(?=\.\w+$)")
_UA = {"User-Agent": "Mozilla/5.0 (book-cover-research; contact patrickcher@gmail.com)"}
_session = requests.Session()
_session.headers.update(_UA)


def canonical_url(url: str) -> str:
    """Drop the `._SX318_` / `._SY475_` downscale suffix goodreads bakes into ~20% of URLs.
    Returns a byte-identical file or a larger master; never smaller."""
    return _SUFFIX_RE.sub("", url)


def _cache_path(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", key)[:120]
    return RAW_CACHE / f"{safe}.img"


def fetch_raw(url: str, cache_key: str | None = None, retries: int = 3, cache: bool = True) -> bytes:
    """`cache=True` keeps a copy under data/covers_raw/ (used by the sample workflow).
    The bulk downloader passes cache=False -- it owns data/covers/<key>.jpg and doesn't
    want a second copy."""
    url = canonical_url(url)
    key = cache_key or hashlib.sha1(url.encode()).hexdigest()
    cp = _cache_path(key)
    if cache and cp.exists():
        return cp.read_bytes()
    if cache:
        RAW_CACHE.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(retries):
        try:
            r = _session.get(url, timeout=25)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                if cache:
                    cp.write_bytes(r.content)
                return r.content
            last = f"HTTP {r.status_code} / {r.headers.get('content-type')}"
        except requests.RequestException as e:  # noqa: PERF203
            last = repr(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"fetch failed for {url}: {last}")


def normalize(img_bytes: bytes, height: int = CANON_H) -> np.ndarray:
    """Decode -> RGB -> resize to fixed height (LANCZOS), preserve aspect. Returns uint8 HxWx3."""
    im = Image.open(io.BytesIO(img_bytes))
    if im.mode != "RGB":
        im = im.convert("RGB")
    w, h = im.size
    if h != height:
        im = im.resize((max(1, round(w * height / h)), height), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def load_cover(src: str, cache_key: str | None = None, height: int = CANON_H) -> np.ndarray:
    """`src` is a goodreads URL or a local file path."""
    p = Path(src)
    if p.exists():
        return normalize(p.read_bytes(), height)
    return normalize(fetch_raw(src, cache_key), height)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Rec. 709 relative luminance, float in [0, 1]."""
    rgb = img.astype(np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722])
