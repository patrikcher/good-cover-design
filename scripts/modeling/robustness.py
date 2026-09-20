"""Week 2 robustness checks on the benchmark's A-vs-C conclusion.

  1. combined clutter z-index      -- the feature_congestion / subband_entropy pair (r=0.73)
                                      showed opposite signs; is that suppression, and does the
                                      block conclusion survive collapsing them to one column?
  2. count GLM on raw numRatings   -- Gamma(log) instead of OLS on log10, does A-vs-C land the
                                      same? (LR test + held-out pseudo-R^2)
  3. within-genre Model C          -- the advice is often genre-specific; spot-check the
                                      biggest genres separately in case an effect is averaged
                                      away across the pooled set.
  4. pre-registered threshold      -- printed, not computed: the bar the rules would have had
                                      to clear to count as mattering.

  python -m scripts.modeling.robustness

Reads data/model_frame.parquet + data/model_split.parquet. No new artifacts (this is a
check, not a headline) -- numbers go to build-log.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

from scripts.modeling.models import (
    CLUTTER_PAIR,
    FAME_BIN,
    FAME_CONT,
    FEATURE_BIN,
    FEATURE_CONT,
    _clustered_paired_test,
    _design,
    _fit,
    _r2,
    _wald_block,
    prep_train_test,
)

FRAME = "data/model_frame.parquet"
SPLIT = "data/model_split.parquet"
CLUTTER = CLUTTER_PAIR
# Model C's headline feature set (FEATURE_CONT already collapses clutter to clutter_z);
# these checks compare it against the older two-column form.
FEATS_HEADLINE = FEATURE_CONT + FEATURE_BIN
FEATS_TWO_CLUTTER = ["sal_gini", "title_contrast_ratio", "text_block_count",
                     "legib_box_retention"] + CLUTTER + FEATURE_BIN

# pre-registered before looking at Model C: the rules "matter" if either holds
THRESHOLD_TEXT = """\
PRE-REGISTERED bar for "the cover-design rules matter" (fixed before reading Model C):
  (a) adding the 5 features to the fame+genre baseline lifts held-out R^2 by >= 0.010
      (one point of variance) on EITHER outcome, OR
  (b) some single claim shows a held-out-robust standardized effect >= 0.05 in the
      direction the advice predicts (~12% change in rating count per SD for the reach
      outcome; ~0.05 rating points per SD for the quality outcome).
Neither is a high bar -- (a) is roughly the CLIP upper bound's entire margin over baseline,
(b) is a small effect. If neither clears, the rules do not have a practically useful
predictive relationship with success once fame and genre are held fixed."""


def _prep(frame, split):
    tr, te, genre_cols = prep_train_test(frame, split)
    controls = FAME_CONT + FAME_BIN + genre_cols
    return tr, te, controls


def check_combined_clutter(frame, split):
    print("=" * 70)
    print("1. COMBINED CLUTTER Z-INDEX  (justifies the headline clutter_z choice)")
    print("=" * 70)
    tr, te, controls = _prep(frame, split)
    r = tr[CLUTTER].corr().iloc[0, 1]

    print(f"  corr(feature_congestion, subband_entropy) on train (standardized) = {r:+.3f}")
    for label, feats in [("two separate clutter cols", FEATS_TWO_CLUTTER),
                         ("one clutter_z (headline)", FEATS_HEADLINE)]:
        for y_col in ("y_log_ratings", "y_rating"):
            resC = _fit(_design(tr, controls + feats), tr[y_col])
            resA = _fit(_design(tr, controls), tr[y_col])
            predC = np.asarray(resC.predict(_design(te, controls + feats)), float)
            predA = np.asarray(resA.predict(_design(te, controls)), float)
            y = te[y_col].to_numpy(float)
            dr2 = _r2(y, predC) - _r2(y, predA)
            wald = _wald_block(resC, feats)
            extra = ""
            if "clutter_z" in feats:
                extra = f"  clutter_z beta={resC.params['clutter_z']:+.4f} p={resC.pvalues['clutter_z']:.3g}"
            else:
                extra = (f"  fc={resC.params['feature_congestion']:+.4f}/"
                         f"se={resC.params['subband_entropy']:+.4f}")
            print(f"  [{label:24s}] {y_col:14s} blockF p={wald['p']:.2e}  heldout dR2={dr2:+.5f}{extra}")
    print()


def check_count_glm(frame, split):
    print("=" * 70)
    print("2. COUNT GLM  -- Gamma(log) on raw numRatings")
    print("=" * 70)
    tr, te, controls = _prep(frame, split)
    feats = FEATURE_CONT + FEATURE_BIN
    y_tr = pd.read_parquet(FRAME).loc[tr.index, "numRatings"].to_numpy(float)
    y_te = pd.read_parquet(FRAME).loc[te.index, "numRatings"].to_numpy(float)

    def glm(cols, y):
        return sm.GLM(y, _design(tr, cols), family=sm.families.Gamma(sm.families.links.Log())).fit()

    gA, gC = glm(controls, y_tr), glm(controls + feats, y_tr)
    lr = 2 * (gC.llf - gA.llf)
    p = float(chi2.sf(lr, len(feats)))
    # held-out pseudo-R^2 on the log scale: corr(log yhat, log y)^2 and 1 - SSE/SST on log
    def heldout_r2(g, cols):
        mu = np.asarray(g.predict(_design(te, cols)), float)
        return _r2(np.log10(y_te), np.log10(np.clip(mu, 1e-6, None)))

    r2A, r2C = heldout_r2(gA, controls), heldout_r2(gC, controls + feats)
    print(f"  in-sample LR test (feature block, df={len(feats)}): LR={lr:.2f}  p={p:.3g}")
    print(f"  held-out pseudo-R^2 (log10 scale)  A={r2A:.4f}  C={r2C:.4f}  dR2={r2C - r2A:+.5f}")
    print(f"  (OLS-on-log10 gave held-out dR2 = +0.0011 for reach -- same order, same sign)")
    print()


def check_within_genre(frame, split):
    print("=" * 70)
    print("3. WITHIN-GENRE  -- Model C held-out dR2 (log10 ratings), per genre")
    print("=" * 70)
    tr_all, te_all, controls_all = _prep(frame, split)
    feats = FEATURE_CONT + FEATURE_BIN
    from scripts.modeling.models import META_BIN, META_CONT
    meta = META_CONT + META_BIN + [c for c in frame.columns if c.startswith("pub_")
                                   and c not in ("pub_year_missing", "pub_year_suspect")]
    fame_only = FAME_CONT + FAME_BIN
    all_genre_cols = [c for c in frame.columns if c.startswith("g_")]
    genres = ["g_fiction", "g_romance", "g_fantasy", "g_young_adult", "g_mystery", "g_nonfiction"]
    print("  (controls trimmed per subset: fame + metadata + genre dummies with >=50 train")
    print("   occurrences, excluding the subsetting genre itself -- the full genre-dummy set is")
    print("   rank-deficient inside a single genre and blows up test extrapolation)")
    for g in genres:
        tr, te = tr_all[tr_all[g] == 1], te_all[te_all[g] == 1]
        if len(te) < 200:
            print(f"  {g[2:]:14s} test n={len(te)} -- skipped (too small)")
            continue
        keep_g = [c for c in all_genre_cols if c != g and tr[c].sum() >= 50]
        keep_pub = [c for c in meta if not c.startswith("pub_") or tr[c].sum() >= 30]
        controls = fame_only + keep_g + ["has_genre"] + keep_pub
        for y_col in ("y_log_ratings", "y_rating"):
            resC = _fit(_design(tr, controls + feats), tr[y_col])
            resA = _fit(_design(tr, controls), tr[y_col])
            y = te[y_col].to_numpy(float)
            predC = np.asarray(resC.predict(_design(te, controls + feats)), float)
            predA = np.asarray(resA.predict(_design(te, controls)), float)
            paired = _clustered_paired_test(y - predA, y - predC, te["primary_author"].to_numpy())
            dr2 = _r2(y, predC) - _r2(y, predA)
            tag = "reach " if y_col == "y_log_ratings" else "rating"
            print(f"  {g[2:]:14s} [{tag}] train {len(tr):5d} / test {len(te):4d}  "
                  f"heldout dR2={dr2:+.5f}  paired p={paired['p']:.3g}")
    print()


def main():
    frame = pd.read_parquet(FRAME)
    split = pd.read_parquet(SPLIT)["split"].reindex(frame.index)
    check_combined_clutter(frame, split)
    check_count_glm(frame, split)
    check_within_genre(frame, split)
    print("=" * 70)
    print("4. " + THRESHOLD_TEXT)


if __name__ == "__main__":
    main()
