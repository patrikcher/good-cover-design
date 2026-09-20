"""Within-author fixed effects: do an author's rule-following covers outperform their own
rule-breaking ones?

The cross-sectional models (models.py) control for author fame with a leave-one-out proxy.
This goes further: it puts an author dummy in the model (via a within-author mean-subtraction),
so *everything* stable about the author is absorbed -- fame, skill, publisher relationship,
fan base, the lot. What is left is variation between one author's own covers.

Restricted to authors with >= 2 books in the working set (~30k books / ~6k authors); solo
authors contribute nothing to a fixed-effects estimate and are dropped. The estimand changes:
this is "within one author's catalogue", not "across all books". Standard errors are clustered
by author. Reported in-sample (the usual way for fixed effects) with a joint test on the
5-feature block; there is no held-out split here -- the author-grouped split used elsewhere
puts each author entirely on one side, which a within-author estimator cannot use.

  python -m scripts.modeling.within_author        # -> stdout + build-log

Reads data/model_frame.parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scripts.modeling.models import (
    CLAIM_TO_COLS,
    CLUTTER_PAIR,
    FEATURE_BIN,
    FEATURE_CONT,
    FEATURE_RAW_CONT,
    META_BIN,
    META_CONT,
    _standardize,
    _wald_block,
)

FRAME = "data/model_frame.parquet"
OUTCOMES = {"y_log_ratings": "reach ", "y_rating": "rating"}


def _demean(df: pd.DataFrame, cols: list[str], group: pd.Series) -> pd.DataFrame:
    g = df.groupby(group)
    return df[cols] - g[cols].transform("mean")


def main() -> None:
    frame = pd.read_parquet(FRAME)
    counts = frame["primary_author"].value_counts()
    multi = frame[frame["primary_author"].map(counts) >= 2].copy()
    n_auth = multi["primary_author"].nunique()
    print(f"within-author FE: {len(multi):,} books / {n_auth:,} authors with >=2 books "
          f"(dropped {len(frame) - len(multi):,} books by solo authors)\n")

    # standardize continuous predictors on this subset, build clutter_z
    cont = FEATURE_RAW_CONT + META_CONT
    z, _ = _standardize(multi, multi, cont)
    for c in cont:
        multi[c] = z[c]
    multi["clutter_z"] = multi[CLUTTER_PAIR].mean(axis=1)

    pub_cols = [c for c in multi.columns if c.startswith("pub_")
               and c not in ("pub_year_missing", "pub_year_suspect")]
    genre_cols = [c for c in multi.columns if c.startswith("g_")]
    feats = FEATURE_CONT + FEATURE_BIN
    # time-varying controls kept (author-constant parts vanish under the within transform anyway)
    controls = META_CONT + META_BIN + pub_cols + genre_cols + ["has_genre"]

    author = multi["primary_author"]
    X_all = _demean(multi, feats + controls, author)
    # drop demeaned columns that are all ~0 (never vary within any author) to keep X full rank
    keep = X_all.columns[(X_all.abs().max() > 1e-9)].tolist()
    X_all = X_all[keep]
    dropped = [c for c in feats + controls if c not in keep]
    if dropped:
        print(f"  {len(dropped)} controls never vary within author, dropped: "
              f"{', '.join(dropped[:8])}{' ...' if len(dropped) > 8 else ''}\n")

    feat_keep = [c for c in feats if c in keep]
    rows = []
    for y_col, tag in OUTCOMES.items():
        y = multi[y_col] - multi.groupby(author)[y_col].transform("mean")
        # residual dof: n - n_authors - k  (author means already removed)
        res = sm.OLS(y.to_numpy(float), X_all.to_numpy(float)).fit(
            cov_type="cluster", cov_kwds={"groups": author.to_numpy()}
        )
        res_df_resid = len(multi) - n_auth - len(keep)
        params = pd.Series(res.params, index=keep)
        bse = pd.Series(res.bse, index=keep)
        pvals = pd.Series(res.pvalues, index=keep)

        block = [c for c in feat_keep]
        # manual cluster-robust Wald F on the feature block
        R = np.zeros((len(block), len(keep)))
        for i, c in enumerate(block):
            R[i, keep.index(c)] = 1.0
        wald = res.wald_test(R, scalar=True, use_f=True)
        print(f"[{tag}] within-author, n={len(multi):,}, clusters={n_auth:,}, "
              f"resid df≈{res_df_resid:,}")
        print(f"   joint block: F({len(block)},{n_auth - 1})={float(wald.statistic):.2f}  "
              f"p={float(wald.pvalue):.3g}")
        for claim, cols in CLAIM_TO_COLS.items():
            c = cols[0]
            if c not in keep:
                continue
            star = "*" if pvals[c] < 0.05 else " "
            print(f"   {star} {claim:22s} {c:20s} beta={params[c]:+.4f}  "
                  f"se={bse[c]:.4f}  p={pvals[c]:.3g}")
            rows.append({"outcome": y_col, "claim": claim, "term": c,
                         "beta_within": float(params[c]), "se": float(bse[c]),
                         "p": float(pvals[c])})
        print()

    out = pd.DataFrame(rows)
    out.to_csv("data/within_author_coefficients.csv", index=False)
    print("wrote data/within_author_coefficients.csv")
    print("\ncompare to the cross-sectional Model C (data/model_c_coefficients.csv):")
    xs = pd.read_csv("data/model_c_coefficients.csv")
    m = out.merge(xs[["outcome", "claim", "beta_per_sd"]], on=["outcome", "claim"])
    for _, r in m.iterrows():
        print(f"   {OUTCOMES[r['outcome']]} {r['claim']:22s} "
              f"cross-sectional {r['beta_per_sd']:+.4f}  within-author {r['beta_within']:+.4f}")


if __name__ == "__main__":
    main()
