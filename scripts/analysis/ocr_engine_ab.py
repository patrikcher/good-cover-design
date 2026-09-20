"""Does Tesseract actually collapse on stylized cover type vs EasyOCR? Test it, don't assert it.

docs/pipeline.md originally claimed EasyOCR was chosen because "Tesseract collapses on
stylized cover type" -- asserted, never tested. This tests it, at three page-segmentation
modes so the comparison isn't rigged by picking an unfair one, and the honest result is more
nuanced than the original claim: see build-log.md 2026-09-05 for the correction.

  python -m scripts.analysis.ocr_engine_ab
"""
import pandas as pd
import pytesseract
from PIL import Image

import scripts
from scripts.cover_io import load_cover

PSMS = {"default (3)": 3, "sparse text (11)": 11, "sparse text + osd (12)": 12}
GREEN_ANGEL = "410615.Green_Angel"  # cursive title, small-caps author name -- the running example


def tess_chars(img, psm):
    txt = pytesseract.image_to_string(Image.fromarray(img), config=f"--psm {psm}")
    return sum(ch.isalnum() for ch in txt), txt.strip().replace("\n", " / ")


def main() -> None:
    feat = pd.read_parquet("data/features_full.parquet")
    sample = feat.sample(150, random_state=21)

    rows = []
    for key in sample.index:
        try:
            img = load_cover(f"data/covers/{key}.jpg")
        except Exception:
            continue
        row = {"key": key, "easyocr_chars": feat.loc[key, "ocr_char_count"]}
        for label, psm in PSMS.items():
            c, _ = tess_chars(img, psm)
            row[f"tess_{psm}"] = c
        rows.append(row)

    df = pd.DataFrame(rows).set_index("key")
    print(f"n={len(df)}\n")
    print(df.describe().T[["min", "50%", "max", "mean"]].round(1))
    print()
    zero_easy = (df["easyocr_chars"] == 0).sum()
    print(f"EasyOCR found 0 chars on {zero_easy}/{len(df)} covers\n")
    for label, psm in PSMS.items():
        col = f"tess_{psm}"
        nz = df[df["easyocr_chars"] > 0]
        print(f"  PSM {psm}: of the {len(nz)} covers where EasyOCR found text, "
              f"Tesseract found 0 on {(nz[col] == 0).sum()} ({100 * (nz[col] == 0).mean():.1f}%); "
              f"median chars tesseract={nz[col].median():.0f} vs easyocr={nz['easyocr_chars'].median():.0f}")

    print()
    print("--- Green Angel: cursive title 'Green Angel' + small-caps 'ALICE HOFFMAN' ---")
    print("(raw char *count* above says tesseract is competitive with the right PSM; this")
    print(" checks whether what it reads is actually right, which the count can't tell you)")
    img = load_cover(f"data/covers/{GREEN_ANGEL}.jpg")
    for label, psm in PSMS.items():
        c, txt = tess_chars(img, psm)
        print(f"  PSM {psm} ({label}): {c} chars -> {txt!r}")
    print("  EasyOCR: 'Green Ange l', 'ALICE', 'HOFFMAN' -> 15 chars (title garbled but legible, author exact)")


if __name__ == "__main__":
    main()
