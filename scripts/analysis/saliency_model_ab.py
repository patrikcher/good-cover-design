"""Was the saliency swap justified? Tested two replacements for spectral residual.

Reproduces the evidence behind build-log.md 2026-09-05, and behind the deployment decision in
scripts/features/saliency.py:
  - the original method (spectral residual) correlates with an "empty background" confound and
    near-duplicates the clutter metric (spearman -0.88 with subband_entropy) -- undermining
    the "5 independent claims" framing.
  - attempt 1, U2Net (salient-OBJECT segmentation): REJECTED. Numbers look like an improvement
    but it returns a near-blank mask on any cover without a discrete photographic/illustrated
    object (text-only, abstract, scenic covers) -- only a visual check catches this, the
    correlation numbers alone would not.
  - attempt 2, DeepGaze IIE (trained on real human eye-tracking data): validated as a real fix
    and **deployed** as the primary sal_gini / sal_top10_mass (spectral residual kept as the
    `_sr` secondary column).

Needs dev-only deps not in the pipeline's requirements-frozen.txt (see requirements-dev.txt for
U2Net's; scripts/features/saliency.py's own docstring / docs/setup.md for DeepGaze's, since
that one is now a real pipeline dependency, not dev-only):
  pip install rembg[cpu]

  python -m scripts.analysis.saliency_model_ab

`example_figures()` rebuilds the exact panels used in the published "Where the Eye Goes"
figure (see notebooks/architecture_choices_eda.ipynb) -- the figure is regenerable from this
module, not a one-off script.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

import scripts
from scripts.cover_io import load_cover, to_gray
from scripts.features.saliency import deepgaze_map, spectral_residual_map as sr_map, _gini

N_SAMPLE = 150
SEED = 7


def flatness(img: np.ndarray) -> float:
    """crude 'empty background' proxy: fraction of pixels close to the image's border color."""
    g = to_gray(img)
    border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
    mode = np.median(border)
    return float((np.abs(g - mode) < 0.06).mean())


def u2net_map(remove, session, img: np.ndarray) -> np.ndarray:
    return np.asarray(remove(Image.fromarray(img), session=session, only_mask=True)).astype(float) / 255


def u2net_gini(remove, session, img: np.ndarray) -> tuple[float, float]:
    mask = u2net_map(remove, session, img)
    k = max(1, int(0.10 * mask.size))
    top10 = float(np.sort(mask.ravel())[-k:].sum() / mask.sum()) if mask.sum() > 0 else 0.0
    return _gini(mask.ravel()), top10


def heat_overlay(img: np.ndarray, heat: np.ndarray, alpha: float = 0.55, renormalize: bool = True) -> Image.Image:
    """Alpha-blend a heatmap (black -> red -> orange -> pale yellow) onto the cover.
    Used for the 'Where the Eye Goes' figure -- kept here so the figure is regenerable from the
    same code that produces the numbers, not a one-off script.

    renormalize=True (spectral residual, DeepGaze): per-image min/max stretch. Safe for these --
    spectral residual is already normalized to [0,1] per image by construction, and a peaked
    density map's max is meaningfully larger than its baseline, so relative display doesn't
    misrepresent it.

    renormalize=False (U2Net): U2Net's mask IS already a meaningful absolute [0,1] confidence
    scale. Per-image min/max stretch would take a mask that's uniformly low-confidence (e.g.
    max 1.0 from a single stray pixel, 99th percentile 0.22 -- i.e. "found nothing") and paint
    it as a confident hot spot. Display it as-is so "found nothing" actually looks like nothing.
    (This was a real bug caught after first publishing the figure -- see build-log.md.)"""
    h = heat.astype(float)
    if renormalize:
        h = (h - h.min()) / (h.max() - h.min() + 1e-9)
    h = np.clip(h, 0, 1) ** 0.6  # gamma-boost mid tones so the hot region reads clearly
    r = np.clip(3.0 * h, 0, 1)
    g = np.clip(3.0 * h - 1.0, 0, 1)
    b = np.clip(3.0 * h - 2.0, 0, 1)
    heat_rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    base = img.astype(float)
    a = (h[..., None] ** 0.8) * alpha
    out = base * (1 - a) + heat_rgb.astype(float) * a
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# The two covers used in the "Where the Eye Goes" figure: one with a discrete illustrated
# figure (U2Net's best case), one text-only with no discrete object (U2Net's failure case).
EXAMPLES = {
    "figurecover": "865850.Epossumondas",
    "textcover": "7090484-a-matter-of-time-book-i",
}


def example_figures() -> dict[str, Image.Image]:
    """Rebuilds the exact panels used in the published 'Where the Eye Goes' figure.
    Returns a flat dict of PIL Images, keys like 'figurecover_plain', 'textcover_old', etc."""
    from rembg import new_session, remove

    u2_session = new_session("u2net")

    out: dict[str, Image.Image] = {}
    for label, key in EXAMPLES.items():
        img = load_cover(f"data/covers/{key}.jpg")
        out[f"{label}_plain"] = Image.fromarray(img)
        out[f"{label}_old"] = heat_overlay(img, sr_map(img))
        out[f"{label}_new"] = heat_overlay(img, deepgaze_map(img))
        if label == "textcover":  # U2Net's failure case is only shown for the text-only cover
            out[f"{label}_u2net"] = heat_overlay(img, u2net_map(remove, u2_session, img), renormalize=False)
    return out


def main() -> None:
    from rembg import new_session, remove

    feat = pd.read_parquet("data/features_full.parquet")
    sample = feat.sample(N_SAMPLE, random_state=SEED)
    u2_session = new_session("u2net")

    rows = []
    for key in sample.index:
        try:
            img = load_cover(f"data/covers/{key}.jpg")
        except Exception:
            continue
        old_gini = _gini(sr_map(img).ravel())
        u2_gini, _ = u2net_gini(remove, u2_session, img)
        deepgaze_gini = _gini(deepgaze_map(img).ravel())
        rows.append({"key": key, "old_gini": old_gini, "u2net_gini": u2_gini,
                     "deepgaze_gini": deepgaze_gini, "flat": flatness(img)})

    df = pd.DataFrame(rows).set_index("key").join(feat[["feature_congestion", "subband_entropy"]])
    print(f"n={len(df)}\n")

    methods = ["old_gini", "u2net_gini", "deepgaze_gini"]
    print(df[methods + ["flat"]].describe().T[["min", "50%", "max", "std"]].round(3))
    print()
    print("spearman vs background-flatness confound (should be near 0):")
    for m in methods:
        print(f"  {m:16s} vs flatness            : {spearmanr(df[m], df.flat).statistic:+.3f}")
    print()
    print("spearman vs clutter (some real overlap expected; near -0.9 means near-duplication):")
    for m in methods:
        print(f"  {m:16s} vs feature_congestion  : {spearmanr(df[m], df.feature_congestion).statistic:+.3f}")
        print(f"  {m:16s} vs subband_entropy     : {spearmanr(df[m], df.subband_entropy).statistic:+.3f}")
    print()
    for m in methods[1:]:
        print(f"spearman(old_gini, {m}) = {spearmanr(df.old_gini, df[m]).statistic:+.3f}  "
              f"(near 0 = unrelated signal, not just a cleaner version of the same one)")


if __name__ == "__main__":
    main()
