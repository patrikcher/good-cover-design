"""Load-bearing check: why the thumbnail-legibility feature was redesigned.

build-log.md (Week 2) claims the first design -- recognized-character retention,
chars_ocr(160px) / chars_ocr(480px) -- was replaced by detector-box retention because v1
was measuring EasyOCR's recognizer floor, not cover legibility. Evidence given: ~13/47
covers scored exactly 0, and v1 had ~0 correlation with title contrast (a real legibility
signal must correlate with contrast). This reproduces both numbers and the A/B table.

  python -m scripts.analysis.thumbnail_legibility_ab

`example_figure()` rebuilds the exact panels used in the published "Reading a Thumbnail"
figure (see notebooks/architecture_choices_eda.ipynb) on a real cover from the full corpus
(`410615.Green_Angel` -- cursive title over an illustration, one of 2,534 covers in
data/features_full.parquet where v1 scores exactly 0 despite real title text being present).
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.stats import spearmanr

from scripts.cover_io import load_cover
from scripts.features.text import (
    _alpha_chars,
    _n_text_boxes,
    _ocr,
    _thumb,
    _title_contrast,
    get_reader,
)

FIGURE_EXAMPLE = "410615.Green_Angel"
FIGURE_UPSCALE = 3  # the 160px thumbnail is tiny; blow it up so boxes are visible in print


def main() -> None:
    rows = []
    for f in sorted(glob.glob("data/sample_covers/*.jpg")):
        img = load_cover(f)
        full = _ocr(img)
        thumb = _thumb(img)
        thumb_d = _ocr(thumb)

        cf = _alpha_chars(full)
        bf = _n_text_boxes(img)
        contrast = _title_contrast(img, full)["title_contrast_ratio"]
        rows.append({
            "key": f.split("/")[-1],
            "chars_full": cf,
            "v1_char_retention": (np.clip(_alpha_chars(thumb_d) / cf, 0, 1) if cf else np.nan),
            "v2_box_retention": (np.clip(_n_text_boxes(thumb) / bf, 0, 1) if bf else np.nan),
            "title_contrast": contrast,
        })
    df = pd.DataFrame(rows).set_index("key")

    v1, v2 = df["v1_char_retention"], df["v2_box_retention"]
    print(f"n = {len(df)} sample covers ({v1.notna().sum()} with any OCR text)\n")
    print(f"  v1 char-retention == 0 exactly : {(v1 == 0).sum()} / {v1.notna().sum()}")
    print(f"  v2 box-retention  == 0 exactly : {(v2 == 0).sum()} / {v2.notna().sum()}")
    print()
    print("  correlation with title contrast (a real legibility signal should be positive):")
    m1 = v1.notna() & df["title_contrast"].notna()
    m2 = v2.notna() & df["title_contrast"].notna()
    print(f"    spearman(v1_char_retention, title_contrast) = {spearmanr(v1[m1], df['title_contrast'][m1]).statistic:+.3f}")
    print(f"    spearman(v2_box_retention,  title_contrast) = {spearmanr(v2[m2], df['title_contrast'][m2]).statistic:+.3f}")
    print()
    print("  spearman(v1, v2) =", f"{spearmanr(v1[v1.notna() & v2.notna()], v2[v1.notna() & v2.notna()]).statistic:+.3f}")
    print()
    print(df.round(3).to_string())


def _detect_boxes(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    horizontal, _free = get_reader().detect(img, text_threshold=0.5, low_text=0.3)
    return [tuple(map(int, b)) for b in horizontal[0]]  # (x_min, x_max, y_min, y_max)


def _draw_boxes(thumb_img: np.ndarray, boxes: list[tuple[int, int, int, int]], color) -> Image.Image:
    big = Image.fromarray(thumb_img).resize(
        (thumb_img.shape[1] * FIGURE_UPSCALE, thumb_img.shape[0] * FIGURE_UPSCALE), Image.LANCZOS
    ).convert("RGB")
    dr = ImageDraw.Draw(big)
    for x0, x1, y0, y1 in boxes:
        dr.rectangle([x0 * FIGURE_UPSCALE, y0 * FIGURE_UPSCALE, x1 * FIGURE_UPSCALE, y1 * FIGURE_UPSCALE],
                     outline=color, width=3)
    return big


def example_figure(key: str = FIGURE_EXAMPLE) -> dict:
    """Returns {'plain', 'v1', 'v2': PIL.Image, 'full_text', 'v1_text': list[str]} for one cover."""
    img = load_cover(f"data/covers/{key}.jpg")
    full_dets = _ocr(img, conf=0.30)
    thumb = _thumb(img)
    thumb_recognized = _ocr(thumb, conf=0.30)  # v1: recognizer output at 160px

    v1_boxes = [(int(d["x0"]), int(d["x1"]), int(d["y0"]), int(d["y1"])) for d in thumb_recognized]
    v2_boxes = _detect_boxes(thumb)

    return {
        "plain": Image.fromarray(thumb).resize(
            (thumb.shape[1] * FIGURE_UPSCALE, thumb.shape[0] * FIGURE_UPSCALE), Image.LANCZOS),
        "v1": _draw_boxes(thumb, v1_boxes, (162, 59, 46)),   # bad/red
        "v2": _draw_boxes(thumb, v2_boxes, (31, 111, 92)),   # good/teal
        "full_text": [d["text"] for d in full_dets],
        "v1_text": [d["text"] for d in thumb_recognized],
        "v1_count": len(v1_boxes),
        "v2_count": len(v2_boxes),
    }


if __name__ == "__main__":
    main()
