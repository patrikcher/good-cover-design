"""Feature-set report: distribution summary + collinearity scan, and (for the sample) a
contact sheet of covers annotated with their features, sorted by Feature Congestion.

  python -m scripts.smoke_features_report                                   # sample + contact sheet
  python -m scripts.smoke_features_report --features data/features_full.parquet --no-sheet

  # want more than 50 covers to eyeball? fetch a bigger (still human-sized) sample first:
  python scripts/fetch_sample.py -n 200 --out data/sample200
  python -m scripts.extract_features --sample-dir data/sample200 --out data/features_sample200.parquet
  python -m scripts.smoke_features_report --covers-dir data/sample200 \
      --features data/features_sample200.parquet --sheet-out data/sample200_contactsheet.png

The collinearity scan is load-bearing: Model C (November) needs a feature set that isn't
badly collinear, and the drop/regularize decision should rest on the full-N numbers, not the
n=50 sample. The contact sheet is a visual sanity check, not an analysis artifact -- it's
capped at MAX_SHEET_ROWS below on purpose; point --features at the full 42k dataset with
--no-sheet, never without it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from scripts.cover_io import load_cover

DEFAULT_COVERS = Path("data/sample_covers")
DEFAULT_SHEET_OUT = Path("data/sample_features_contactsheet.png")
TILE_W, TILE_H, PAD, COLS = 200, 300, 4, 8
MAX_SHEET_ROWS = 500  # a contact sheet is for eyeballing by hand -- past this it's not one


def stats(df: pd.DataFrame) -> None:
    num = df.select_dtypes("number")
    print(f"\n--- feature summary (n={len(df)}) ---")
    print(num.describe(percentiles=[.1, .5, .9]).T[["min", "10%", "50%", "90%", "max", "std"]].round(3))
    print("\nmissing:", {k: int(v) for k, v in num.isna().sum().items() if v})
    print("\n--- |spearman| > 0.6 feature pairs (collinearity watch for the Nov models) ---")
    c = num.corr("spearman").abs()
    for a in c.columns:
        for b in c.columns:
            if a < b and c.loc[a, b] > 0.6:
                print(f"  {a:26s} {b:26s} {c.loc[a, b]:.2f}")


def contact_sheet(df: pd.DataFrame, covers_dir: Path, sheet_out: Path) -> None:
    if len(df) > MAX_SHEET_ROWS:
        raise SystemExit(
            f"contact_sheet: {len(df)} rows exceeds MAX_SHEET_ROWS ({MAX_SHEET_ROWS}) -- "
            "this isn't a human-reviewable sheet anymore. Pass --no-sheet, or point "
            "--features at a smaller sample."
        )
    df = df.sort_values("feature_congestion")
    files = {p.stem: p for p in covers_dir.glob("*.jpg")}
    rows = int(np.ceil(len(df) / COLS))
    sheet = Image.new("RGB", (COLS * (TILE_W + PAD), rows * (TILE_H + PAD)), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (key, r) in enumerate(df.iterrows()):
        if key not in files:
            continue
        cx, cy = (i % COLS) * (TILE_W + PAD), (i // COLS) * (TILE_H + PAD)
        im = Image.fromarray(load_cover(str(files[key])))
        im.thumbnail((TILE_W, TILE_H - 46))
        sheet.paste(im, (cx + (TILE_W - im.width) // 2, cy))
        tc = r["title_contrast_ratio"]
        cap = (f"FC {r['feature_congestion']:.1f}  SE {r['subband_entropy']:.1f}\n"
               f"gini {r['sal_gini']:.2f}  blocks {int(r['text_block_count'])}\n"
               + (f"contrast {tc:.1f}  legib {r['legib_box_retention']:.2f}"
                  if pd.notna(tc) else f"(no OCR text)  legib {r['legib_box_retention']:.2f}"))
        dr.multiline_text((cx + 3, cy + TILE_H - 44), cap, fill="black", spacing=2)
    sheet.save(sheet_out)
    print(f"wrote {sheet_out}  ({sheet.size[0]}x{sheet.size[1]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features_sample.parquet")
    ap.add_argument("--covers-dir", default=str(DEFAULT_COVERS),
                     help="folder of cover jpgs to draw the contact sheet from")
    ap.add_argument("--sheet-out", default=str(DEFAULT_SHEET_OUT))
    ap.add_argument("--no-sheet", action="store_true", help="skip the contact sheet (needs sample covers)")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    if not args.no_sheet:
        contact_sheet(df, Path(args.covers_dir), Path(args.sheet_out))
    stats(df)


if __name__ == "__main__":
    main()
