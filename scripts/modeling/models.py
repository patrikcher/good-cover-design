"""The November benchmark: four models x two outcomes on the fixed author-grouped split.

  Model A  baseline    : genre + author-fame            -> outcome
  Model B  cover-only   : the 5 claim features           -> outcome   (naive/confounded picture)
  Model C  controlled   : A + the 5 claim features       -> outcome   (THE test: A vs C)
  Model D  CLIP         : A + PCA-reduced 512-d embedding -> outcome   (upper bound, vs C)

Outcomes are kept separate and reported separately:
  y_log_ratings = log10(numRatings)   (reach / exposure)
  y_rating      = average rating       (reception among those who read it)

The A-vs-C test is a robust (HC3) Wald F-test on the joint nullity of the feature block, fit
on train, plus a held-out check: delta R^2 / delta RMSE on the test set with an author-
clustered paired test on squared errors and a bootstrap CI over test authors.

  python -m scripts.modeling.models              # -> data/model_results.json, data/model_c_coefficients.csv

Requires data/model_frame.parquet (scripts.modeling.assemble) and data/model_split.parquet
(scripts.modeling.split). CLIP from data/clip_embeddings.npz.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA

FRAME = "data/model_frame.parquet"
SPLIT = "data/model_split.parquet"
# Model D is the "ceiling" -- an upper bound on any cover signal -- so it uses the larger
# ViT-L-14 embedding (data/clip_embeddings_l14.npz), not the ViT-B-32 that Part 1's pipeline
# ships as the standard catch-all. Override with CLIP_NPZ to compare. See build-log 2026-09-06.
CLIP = os.environ.get(
    "CLIP_NPZ",
    "data/clip_embeddings_l14.npz" if os.path.exists("data/clip_embeddings_l14.npz")
    else "data/clip_embeddings.npz",
)
RESULTS_JSON = "data/model_results.json"
COEF_CSV = "data/model_c_coefficients.csv"

SEED = 42
N_BOOT = 2000
CLIP_PCS = 100

OUTCOMES = ["y_log_ratings", "y_rating"]

FAME_CONT = ["fame_log_sum_ratings", "fame_log_max_ratings", "fame_log_n_books", "fame_mean_rating"]
# fame_mean_rating_missing is identical to author_is_solo (mean_rating is NaN iff solo) -> use one
FAME_BIN = ["author_is_solo"]

# Metadata controls added 2026-09-06 (build-log): publication era, series, publisher. These
# join Model A/C/D as extra controls -- the point is to check the cover coefficients survive
# them (esp. focal point: old covers are busier AND have more accumulated ratings).
META_CONT = ["book_age", "log_series_size"]
META_BIN = ["pub_year_missing", "pub_year_suspect", "has_series", "self_published",
            "publisher_missing"]

# The two Rosenholtz clutter columns correlate 0.72 and, entered separately, split into an
# equal-and-opposite suppression pair (build-log Week 2). "Low clutter" is one claim -> one
# column: clutter_z = mean of the two standardized values.
CLUTTER_PAIR = ["feature_congestion", "subband_entropy"]
FEATURE_RAW_CONT = ["sal_gini", "title_contrast_ratio", "text_block_count",
                    "legib_box_retention"] + CLUTTER_PAIR       # what gets standardized
FEATURE_CONT = ["sal_gini", "title_contrast_ratio", "text_block_count",
                "legib_box_retention", "clutter_z"]             # what enters the models
FEATURE_BIN = ["title_contrast_ratio_missing", "legib_box_retention_missing"]

# claim -> the feature column(s) that carry it, for the per-claim readout
CLAIM_TO_COLS = {
    "single_focal_point": ["sal_gini"],
    "text_bg_contrast": ["title_contrast_ratio"],
    "typographic_simplicity": ["text_block_count"],
    "thumbnail_legibility": ["legib_box_retention"],
    "low_clutter": ["clutter_z"],
}


def _standardize(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1.0)
    return (train[cols] - mu) / sd, (test[cols] - mu) / sd


def prep_train_test(frame: pd.DataFrame, split: pd.Series):
    """Split, standardize continuous cols on TRAIN stats, build clutter_z.

    Returns (train, test, ctrl_cols) where ctrl_cols is the full non-fame control block:
    genre dummies + has_genre + book age / series / publisher metadata. Shared by models.py
    and the check scripts so every model sees an identically-prepared frame.
    """
    train, test = frame[split == "train"].copy(), frame[split == "test"].copy()
    cont_cols = FAME_CONT + FEATURE_RAW_CONT + META_CONT
    ztr, zte = _standardize(train, test, cont_cols)
    for c in cont_cols:
        train[c], test[c] = ztr[c], zte[c]
    train["clutter_z"] = train[CLUTTER_PAIR].mean(axis=1)
    test["clutter_z"] = test[CLUTTER_PAIR].mean(axis=1)
    pub_cols = [c for c in frame.columns if c.startswith("pub_") and c != "pub_year_missing"
                and c != "pub_year_suspect"]
    genre_cols = ([c for c in frame.columns if c.startswith("g_")] + ["has_genre"]
                  + META_CONT + META_BIN + pub_cols)
    return train, test, genre_cols


def _design(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return sm.add_constant(df[cols], has_constant="add")


def _r2(y, yhat) -> float:
    y = np.asarray(y, float)
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((y - np.asarray(yhat, float)) ** 2).sum()
    return 1.0 - ss_res / ss_tot


def _rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def _fit(Xtr, ytr):
    return sm.OLS(ytr.to_numpy(float), Xtr).fit(cov_type="HC3")


def _wald_block(res, block_cols: list[str]):
    """Robust joint F-test that every coefficient in block_cols is zero."""
    present = [c for c in block_cols if c in res.params.index]
    test = res.wald_test(present, scalar=True, use_f=True)
    return {"F": float(test.statistic), "p": float(test.pvalue),
            "df_num": len(present), "df_denom": int(res.df_resid)}


def _boot_delta_r2(y, pred_a, pred_c, authors, rng, n=N_BOOT):
    """Bootstrap delta R^2 (C - A) resampling whole test authors with replacement."""
    y = np.asarray(y, float)
    by_author: dict = {}
    for i, a in enumerate(authors):
        by_author.setdefault(a, []).append(i)
    groups = [np.array(v) for v in by_author.values()]
    out = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[j] for j in pick])
        out[b] = _r2(y[idx], pred_c[idx]) - _r2(y[idx], pred_a[idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _clustered_paired_test(err_a, err_c, authors):
    """Paired test on squared-error reduction (A - C), averaged within test author."""
    d = np.asarray(err_a, float) ** 2 - np.asarray(err_c, float) ** 2
    s = pd.Series(d).groupby(np.asarray(authors)).mean()
    t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
    from scipy.stats import t as tdist

    p = float(2 * tdist.sf(abs(t), len(s) - 1))
    return {"mean_sqerr_reduction": float(s.mean()), "t": float(t), "p": p, "n_authors": int(len(s))}


def run_outcome(frame: pd.DataFrame, split: pd.Series, clip_pcs: pd.DataFrame, y_col: str, rng):
    train, test, genre_cols = prep_train_test(frame, split)
    controls = FAME_CONT + FAME_BIN + genre_cols
    features = FEATURE_CONT + FEATURE_BIN

    specs = {
        "A": controls,
        "B": features,
        "C": controls + features,
        "D": controls + list(clip_pcs.columns),
    }
    # attach CLIP PCs (already aligned by key, standardized on train inside build)
    train = train.join(clip_pcs, how="left")
    test = test.join(clip_pcs, how="left")

    ytr, yte = train[y_col], test[y_col]
    authors_te = test["primary_author"].to_numpy()

    fits, rows = {}, {}
    for name, cols in specs.items():
        Xtr, Xte = _design(train, cols), _design(test, cols)
        res = _fit(Xtr, ytr)
        fits[name] = (res, cols)
        pred_te = res.predict(Xte)
        rows[name] = {
            "n_params": int(len(cols)) + 1,
            "r2_train": float(res.rsquared),
            "r2_test": _r2(yte, pred_te),
            "rmse_test": _rmse(yte, pred_te),
        }

    # --- A vs C : robust joint block test (train), held-out delta, bootstrap, paired test ---
    resC, _ = fits["C"]
    block_test = _wald_block(resC, features)

    predA = np.asarray(fits["A"][0].predict(_design(test, specs["A"])), float)
    predC = np.asarray(fits["C"][0].predict(_design(test, specs["C"])), float)
    predD = np.asarray(fits["D"][0].predict(_design(test, specs["D"])), float)
    yte_arr = yte.to_numpy(float)

    d_r2_ci = _boot_delta_r2(yte_arr, predA, predC, authors_te, rng)
    paired = _clustered_paired_test(yte_arr - predA, yte_arr - predC, authors_te)

    a_vs_c = {
        "block_wald_f_train": block_test,
        "delta_r2_test": rows["C"]["r2_test"] - rows["A"]["r2_test"],
        "delta_r2_test_ci95": d_r2_ci,
        "delta_rmse_test": rows["C"]["rmse_test"] - rows["A"]["rmse_test"],
        "paired_sqerr_test": paired,
    }
    b_vs_c = {
        "r2_test_B": rows["B"]["r2_test"],
        "r2_test_C": rows["C"]["r2_test"],
        "note": "B has no controls; gap B->C is how much of B's fit was genre/fame leaking through the cover",
    }
    d_vs_c = {
        "delta_r2_test_D_minus_C": rows["D"]["r2_test"] - rows["C"]["r2_test"],
        "delta_r2_test_D_minus_A": rows["D"]["r2_test"] - rows["A"]["r2_test"],
        "paired_sqerr_D_vs_C": _clustered_paired_test(yte_arr - predC, yte_arr - predD, authors_te),
    }

    # --- per-claim coefficients from Model C (HC3) ---
    coef_rows = []
    ci = resC.conf_int()
    for claim, cols in CLAIM_TO_COLS.items():
        for c in cols:
            coef_rows.append({
                "outcome": y_col, "claim": claim, "term": c,
                "beta_per_sd": float(resC.params[c]),
                "se": float(resC.bse[c]),
                "ci_lo": float(ci.loc[c, 0]), "ci_hi": float(ci.loc[c, 1]),
                "p": float(resC.pvalues[c]),
            })

    return {"models": rows, "a_vs_c": a_vs_c, "b_vs_c": b_vs_c, "d_vs_c": d_vs_c}, coef_rows


def build_clip_pcs(keys: pd.Index, split: pd.Series) -> pd.DataFrame:
    z = np.load(CLIP, allow_pickle=True)
    emb = pd.DataFrame(z["embeddings"], index=pd.Index(z["keys"], name="key"))
    emb = emb.reindex(keys)
    missing = emb.isna().any(axis=1).sum()
    if missing:
        emb = emb.fillna(emb.mean())  # a handful of covers with no embedding
    tr = split == "train"
    mu, sd = emb[tr].mean(), emb[tr].std().replace(0, 1.0)
    embz = (emb - mu) / sd
    pca = PCA(n_components=CLIP_PCS, random_state=SEED).fit(embz[tr].to_numpy())
    pcs = pca.transform(embz.to_numpy())
    evr = float(pca.explained_variance_ratio_.sum())
    cols = [f"clip_pc{i:02d}" for i in range(CLIP_PCS)]
    out = pd.DataFrame(pcs, index=keys, columns=cols)
    out.attrs["evr"] = evr
    out.attrs["n_missing_embed"] = int(missing)
    return out


def main() -> None:
    frame = pd.read_parquet(FRAME)
    split = pd.read_parquet(SPLIT)["split"].reindex(frame.index)
    rng = np.random.default_rng(SEED)

    clip_pcs = build_clip_pcs(frame.index, split)
    print(f"CLIP: {CLIP_PCS} PCs retain {clip_pcs.attrs['evr']:.1%} of variance "
          f"({clip_pcs.attrs['n_missing_embed']} covers had no embedding, mean-filled)\n")

    results, all_coefs = {}, []
    for y in OUTCOMES:
        res, coefs = run_outcome(frame, split, clip_pcs, y, rng)
        results[y] = res
        all_coefs.extend(coefs)

        print(f"================ {y} ================")
        m = res["models"]
        print(f"{'model':6s} {'k':>4s} {'R2 train':>9s} {'R2 test':>9s} {'RMSE test':>10s}")
        for name in ["A", "B", "C", "D"]:
            r = m[name]
            print(f"{name:6s} {r['n_params']:4d} {r['r2_train']:9.4f} {r['r2_test']:9.4f} {r['rmse_test']:10.4f}")
        avc = res["a_vs_c"]
        bt = avc["block_wald_f_train"]
        print(f"\n  A vs C  -- feature block, robust Wald F (train): "
              f"F({bt['df_num']},{bt['df_denom']})={bt['F']:.3f}  p={bt['p']:.3g}")
        lo, hi = avc["delta_r2_test_ci95"]
        print(f"          held-out delta R2 = {avc['delta_r2_test']:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]")
        print(f"          held-out delta RMSE = {avc['delta_rmse_test']:+.5f}")
        p = avc["paired_sqerr_test"]
        print(f"          author-clustered paired test on sq-err reduction: "
              f"t={p['t']:.3f}  p={p['p']:.3g}  (n={p['n_authors']} authors)")
        dvc = res["d_vs_c"]
        print(f"\n  D vs C  held-out delta R2 (D - C) = {dvc['delta_r2_test_D_minus_C']:+.5f}"
              f"   (D - A = {dvc['delta_r2_test_D_minus_A']:+.5f})")
        print(f"  B vs C  R2 test  B={res['b_vs_c']['r2_test_B']:.4f}  C={res['b_vs_c']['r2_test_C']:.4f}")
        print()
        print("  Model C per-claim coefficients (per 1 SD, HC3):")
        for row in coefs:
            star = "*" if row["p"] < 0.05 else " "
            print(f"   {star} {row['claim']:22s} {row['term']:22s} "
                  f"beta={row['beta_per_sd']:+.4f}  CI[{row['ci_lo']:+.4f},{row['ci_hi']:+.4f}]  p={row['p']:.3g}")
        print()

    with open(RESULTS_JSON, "w") as f:
        json.dump({"clip_evr": clip_pcs.attrs["evr"], "n_clip_pcs": CLIP_PCS,
                   "results": results}, f, indent=2)
    pd.DataFrame(all_coefs).to_csv(COEF_CSV, index=False)
    print(f"wrote {RESULTS_JSON} and {COEF_CSV}")


if __name__ == "__main__":
    main()
