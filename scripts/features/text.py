"""Three claims, one OCR pass each at two resolutions:

  * "high text/background contrast"      -> title_contrast_ratio (WCAG), ink-vs-paper inside the title box
  * "two-font maximum / simplicity"      -> text_block_count (merged OCR clusters; rough proxy)
  * "thumbnail legibility"               -> legib_char_retention at 160 px tall vs full (480 px)

OCR engine: EasyOCR (CRAFT detector + CRNN recognizer). CPU, English. Picked over Tesseract --
tested, not just asserted, see scripts/analysis/ocr_engine_ab.py and build-log.md 2026-09-05.
The honest finding is narrower than "Tesseract collapses": with the default page-segmentation
mode it does (misses text entirely on ~32% of covers EasyOCR reads); tuned to a fairer
sparse-text mode its raw recall is competitive. What doesn't hold up under tuning is quality --
on a cursive/decorative title, tuned Tesseract fragments it into unreadable noise ("Gave.",
"L)", ":") rather than words, which would corrupt block-count and title-region measurements
even where the character count looks fine.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu

from scripts.cover_io import to_gray

_CONF = 0.30          # keep detections at/above this recognizer confidence
_THUMB_H = 160        # simulated browse-thumbnail height
_reader = None


def get_reader():
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def _ocr(img: np.ndarray, conf: float = _CONF) -> list[dict]:
    out = []
    for poly, text, c in get_reader().readtext(img):
        if c < conf or not text.strip():
            continue
        p = np.asarray(poly, dtype=np.float64)
        x0, y0 = p[:, 0].min(), p[:, 1].min()
        x1, y1 = p[:, 0].max(), p[:, 1].max()
        out.append({
            "text": text.strip(), "conf": float(c),
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "h": y1 - y0, "w": x1 - x0, "area": (x1 - x0) * (y1 - y0),
        })
    return out


# ---- claim: high text/background contrast -------------------------------------------------
def _title_contrast(img: np.ndarray, dets: list[dict]) -> dict[str, float]:
    if not dets:
        return {"title_contrast_ratio": np.nan, "median_text_contrast_ratio": np.nan}

    lum = to_gray(img)  # linearized relative luminance in [0, 1]
    ratios = []
    for d in dets:
        y0, y1 = int(max(0, d["y0"])), int(min(lum.shape[0], d["y1"]))
        x0, x1 = int(max(0, d["x0"])), int(min(lum.shape[1], d["x1"]))
        crop = lum[y0:y1, x0:x1]
        if crop.size < 25 or np.ptp(crop) < 1e-4:
            continue
        try:
            t = threshold_otsu(crop)
        except ValueError:
            continue
        dark, light = crop[crop <= t], crop[crop > t]
        if dark.size == 0 or light.size == 0:
            continue
        l_hi, l_lo = float(light.mean()), float(dark.mean())
        ratios.append(((l_hi + 0.05) / (l_lo + 0.05), d["area"]))

    if not ratios:
        return {"title_contrast_ratio": np.nan, "median_text_contrast_ratio": np.nan}
    title = max(ratios, key=lambda r: r[1])[0]
    med = float(np.median([r[0] for r in ratios]))
    return {"title_contrast_ratio": float(title), "median_text_contrast_ratio": med}


# ---- claim: two-font maximum / typographic simplicity ------------------------------------
def _block_count(dets: list[dict]) -> int:
    """Merge detections into blocks: union-find over boxes that are vertically adjacent
    (gap < 0.8x line height) and horizontally overlapping. Count of blocks is the proxy."""
    n = len(dets)
    if n <= 1:
        return n
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            a, b = dets[i], dets[j]
            x_ov = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
            lh = max(a["h"], b["h"])
            v_gap = max(a["y0"], b["y0"]) - min(a["y1"], b["y1"])
            if x_ov > -0.3 * min(a["w"], b["w"]) and -0.5 * lh <= v_gap <= 0.8 * lh:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


# ---- claim: thumbnail legibility -------------------------------------------------------
# FIRST DESIGN (replaced): recognized-character retention -- chars_ocr(thumb) / chars_ocr(full).
# On the Week-2 smoke test this was dominated by EasyOCR's *recognizer* floor at 160 px
# (13/47 covers scored exactly 0, and it had ~0 correlation with contrast -- which a real
# legibility signal must have). It measured "does CRNN fire at 30 px cap-height", not the cover.
# See build-log.md 2026-09-04.
#
# CURRENT DESIGN: text-region *detector* retention -- CRAFT degrades gracefully, so
# n_boxes(thumb)/n_boxes(full) tracks "can you still see there's a title" rather than
# "can a model read it". char retention is kept as a secondary, transparently-noisy column.

def _alpha_chars(dets: list[dict]) -> int:
    return sum(sum(ch.isalnum() for ch in d["text"]) for d in dets)


def _n_text_boxes(img: np.ndarray) -> int:
    horizontal, free = get_reader().detect(img, text_threshold=0.5, low_text=0.3)
    return len(horizontal[0]) + len(free[0])


def _thumb(img: np.ndarray, height: int = _THUMB_H) -> np.ndarray:
    w = max(1, round(img.shape[1] * height / img.shape[0]))
    return np.asarray(Image.fromarray(img).resize((w, height), Image.LANCZOS))


def text_features(img: np.ndarray) -> dict[str, float]:
    full = _ocr(img)
    thumb_img = _thumb(img)
    thumb_d = _ocr(thumb_img)

    boxes_full = _n_text_boxes(img)
    boxes_thumb = _n_text_boxes(thumb_img)

    cf, ct = _alpha_chars(full), _alpha_chars(thumb_d)
    conf_full = np.mean([d["conf"] for d in full]) if full else np.nan
    conf_thumb = np.mean([d["conf"] for d in thumb_d]) if thumb_d else np.nan

    feats = {
        "text_block_count": _block_count(full),
        "n_text_detections": len(full),
        "ocr_char_count": cf,
        "n_text_boxes_full": boxes_full,
        # primary thumbnail-legibility feature
        "legib_box_retention": float(np.clip(boxes_thumb / boxes_full, 0, 1)) if boxes_full else np.nan,
        # secondary, kept for transparency -- known noisy
        "legib_char_retention": float(np.clip(ct / cf, 0, 1)) if cf else np.nan,
        "legib_conf_ratio": float(conf_thumb / conf_full) if full and thumb_d else np.nan,
    }
    feats.update(_title_contrast(img, full))
    return feats
