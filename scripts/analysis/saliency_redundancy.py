"""Load-bearing check: why the saliency feature set has two columns, not three.

Claimed in build-log.md (Week 2): normalized Shannon entropy of the spectral-residual
saliency map was dropped because it is spearman -1.0 with `sal_gini` on the sample -- a pure
monotone restatement, no new information. This script reproduces that number so the decision
is auditable.

  python -m scripts.analysis.saliency_redundancy
"""
from __future__ import annotations

import glob

import numpy as np
from scipy.stats import spearmanr

from scripts.cover_io import load_cover
from scripts.features.saliency import _gini, saliency_map


def _norm_entropy(sal: np.ndarray) -> float:
    p = sal.ravel()
    s = p.sum()
    p = p / s if s > 0 else np.full_like(p, 1 / p.size)
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(p.size))


def main() -> None:
    files = sorted(glob.glob("data/sample_covers/*.jpg"))
    gini, ent, top10 = [], [], []
    for f in files:
        sal = saliency_map(load_cover(f))
        flat = sal.ravel()
        gini.append(_gini(flat))
        ent.append(_norm_entropy(sal))
        k = max(1, int(0.10 * flat.size))
        top10.append(np.sort(flat)[-k:].sum() / flat.sum())

    g, e, t = np.array(gini), np.array(ent), np.array(top10)
    print(f"n = {len(files)} sample covers\n")
    print(f"  spearman(sal_gini, sal_norm_entropy) = {spearmanr(g, e).statistic:+.4f}   <- reason entropy was dropped")
    print(f"  pearson (sal_gini, sal_norm_entropy) = {np.corrcoef(g, e)[0, 1]:+.4f}")
    print(f"  spearman(sal_gini, sal_top10_mass)   = {spearmanr(g, t).statistic:+.4f}   <- 0.94, kept: not a restatement")
    print()
    print("  interpretation: after smoothing + [0,1] normalization the saliency maps vary along")
    print("  essentially one axis (peaky vs flat), so every concentration statistic ranks the")
    print("  covers identically. gini + top10 mass are kept; entropy adds nothing.")


if __name__ == "__main__":
    main()
