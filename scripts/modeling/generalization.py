"""Week 3 pre-writeup checks: does the A-vs-C null generalize?

  1. low-fame subset      -- the advice is really for authors with no reputation. Re-run
                             Model C on solo-author and low-fame books, and test a
                             feature x author_is_solo interaction. If covers matter anywhere,
                             it's here.
  2. grouped k-fold       -- the headline delta R^2 is one 80/20 author-grouped split.
                             Repeat over 5 GroupKFold folds; is the sign stable?
  3. nonlinearity         -- everything entered linearly, but the advice is about thresholds.
                             Add quadratic terms for the two features with real coefficients
                             (sal_gini, legib_box_retention) and joint-test them; plus a
                             decile table for sal_gini.

  python -m scripts.modeling.generalization

Reads data/model_frame.parquet + data/model_split.parquet. No new artifacts -- numbers go to
build-log.md.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from scripts.modeling.models import (
    CLUTTER_PAIR,
    FAME_BIN,
    FAME_CONT,
    FEATURE_BIN,
    FEATURE_CONT,
    FEATURE_RAW_CONT,
    META_BIN,
    META_CONT,
    _clustered_paired_test,
    _design,
    _fit,
    _r2,
    _standardize,
    _wald_block,
    prep_train_test,
)

FRAME = "data/model_frame.parquet"
SPLIT = "data/model_split.parquet"
OUTCOMES = ["y_log_ratings", "y_rating"]
FEATS = FEATURE_CONT + FEATURE_BIN


def _prep_pair(train, test):
    train, test = train.copy(), test.copy()
    cont = FAME_CONT + FEATURE_RAW_CONT + META_CONT
    ztr, zte = _standardize(train, test, cont)
    for c in cont:
        train[c], test[c] = ztr[c], zte[c]
    train["clutter_z"] = train[CLUTTER_PAIR].mean(axis=1)
    test["clutter_z"] = test[CLUTTER_PAIR].mean(axis=1)
    pub_cols = [c for c in train.columns if c.startswith("pub_")
                and c not in ("pub_year_missing", "pub_year_suspect")]
    ctrl = (FAME_CONT + FAME_BIN + [c for c in train.columns if c.startswith("g_")]
            + ["has_genre"] + META_CONT + META_BIN + pub_cols)
    return train, test, ctrl


def _delta_r2(train, test, y_col, controls, feats=FEATS):
    resA = _fit(_design(train, controls), train[y_col])
    resC = _fit(_design(train, controls + feats), train[y_col])
    y = test[y_col].to_numpy(float)
    pA = np.asarray(resA.predict(_design(test, controls)), float)
    pC = np.asarray(resC.predict(_design(test, controls + feats)), float)
    return _r2(y, pC) - _r2(y, pA), resA, resC, pA, pC


def check_low_fame(frame, split):
    print("=" * 70)
    print("1. LOW-FAME SUBSET  -- does the cover matter when the author is unknown?")
    print("=" * 70)
    train, test, gcols = prep_train_test(frame, split)
    controls = FAME_CONT + FAME_BIN + gcols

    # author_is_solo is 0/1 (not standardized); fame_log_sum_ratings is standardized here
    solo_tr = train["author_is_solo"].to_numpy().astype(bool)
    solo_te = test["author_is_solo"].to_numpy().astype(bool)
    cut = train.loc[~solo_tr, "fame_log_sum_ratings"].quantile(0.33)
    low_tr = (~solo_tr) & (train["fame_log_sum_ratings"].to_numpy() <= cut)
    low_te = (~solo_te) & (test["fame_log_sum_ratings"].to_numpy() <= cut)

    for name, m_tr, m_te in [
        ("solo authors      ", solo_tr, solo_te),
        ("low-fame (non-solo)", low_tr, low_te),
        ("solo OR low-fame   ", solo_tr | low_tr, solo_te | low_te),
    ]:
        tr, te = train[m_tr], test[m_te]
        for y_col in OUTCOMES:
            dr2, _, resC, pA, pC = _delta_r2(tr, te, y_col, controls)
            y = te[y_col].to_numpy(float)
            paired = _clustered_paired_test(y - pA, y - pC, te["primary_author"].to_numpy())
            tag = "reach " if y_col == OUTCOMES[0] else "rating"
            print(f"  {name} [{tag}] train {len(tr):5d} / test {len(te):4d}  "
                  f"heldout dR2={dr2:+.5f}  paired p={paired['p']:.3g}")
    print()

    # feature x author_is_solo interaction on the full training set
    print("  feature x author_is_solo interaction (full train, joint Wald on the 5 products):")
    tr = train.copy()
    solo_full = tr["author_is_solo"].to_numpy().astype(float)
    inter_cols = []
    for c in FEATURE_CONT:
        ic = f"{c}__x_solo"
        tr[ic] = tr[c].to_numpy() * solo_full
        inter_cols.append(ic)
    for y_col in OUTCOMES:
        res = _fit(_design(tr, controls + FEATS + inter_cols), tr[y_col])
        w = _wald_block(res, inter_cols)
        tag = "reach " if y_col == OUTCOMES[0] else "rating"
        print(f"    [{tag}] joint Wald F({w['df_num']},{w['df_denom']})={w['F']:.2f}  p={w['p']:.3g}")
    print()


def check_kfold(frame, split):
    print("=" * 70)
    print("2. GROUPED 5-FOLD  -- is the held-out delta R^2 stable across splits?")
    print("=" * 70)
    gkf = GroupKFold(n_splits=5)
    groups = frame["primary_author"]
    per = {y: [] for y in OUTCOMES}
    for k, (tr_i, te_i) in enumerate(gkf.split(frame, groups=groups)):
        tr, te, controls = _prep_pair(frame.iloc[tr_i], frame.iloc[te_i])
        for y_col in OUTCOMES:
            dr2, *_ = _delta_r2(tr, te, y_col, controls)
            per[y_col].append(dr2)
    out = {}
    for y_col in OUTCOMES:
        v = np.array(per[y_col])
        tag = "reach " if y_col == OUTCOMES[0] else "rating"
        stable = bool((v > 0).all() or (v < 0).all())
        print(f"  [{tag}] dR2 per fold: [{', '.join(f'{x:+.5f}' for x in v)}]")
        print(f"          mean {v.mean():+.5f}  sd {v.std(ddof=1):.5f}  (sign stable: {stable})")
        out[y_col] = {"per_fold": v.tolist(), "mean": float(v.mean()),
                      "sd": float(v.std(ddof=1)), "sign_stable": stable}
    print()
    return out


def check_nonlinearity(frame, split):
    print("=" * 70)
    print("3. NONLINEARITY  -- quadratic terms for sal_gini & legib_box_retention")
    print("=" * 70)
    train, test, gcols = prep_train_test(frame, split)
    controls = FAME_CONT + FAME_BIN + gcols
    quad = ["sal_gini", "legib_box_retention"]
    tr = train.copy()
    q_cols = []
    for c in quad:
        tr[f"{c}_sq"] = tr[c].to_numpy() ** 2
        q_cols.append(f"{c}_sq")
    for y_col in OUTCOMES:
        res = _fit(_design(tr, controls + FEATS + q_cols), tr[y_col])
        w = _wald_block(res, q_cols)
        tag = "reach " if y_col == OUTCOMES[0] else "rating"
        betas = "  ".join(f"{c}={res.params[c]:+.4f}(p={res.pvalues[c]:.2g})" for c in q_cols)
        print(f"  [{tag}] joint Wald on quadratics F({w['df_num']},{w['df_denom']})={w['F']:.2f} "
              f"p={w['p']:.3g}   {betas}")
    print()

    # decile table: outcome residualized on controls, mean by sal_gini decile
    print("  sal_gini decile vs mean residual reach (outcome residualized on A controls):")
    resA = _fit(_design(train, controls), train["y_log_ratings"])
    resid = train["y_log_ratings"].to_numpy() - np.asarray(resA.predict(_design(train, controls)), float)
    dec = pd.qcut(train["sal_gini"], 10, labels=False)
    tab = pd.DataFrame({"resid": resid, "dec": dec}).groupby("dec")["resid"].mean()
    print("   " + "  ".join(f"d{i}:{v:+.3f}" for i, v in tab.items()))
    print("   (monotone decreasing would mean 'more concentration -> less reach' is real & linear;")
    print("    a U or flat middle would mean the linear coefficient is misleading)")
    print()


def check_modern_subset(frame, split):
    print("=" * 70)
    print("4. MODERN SUBSET  -- books published 1995+ (book_age <= 26)")
    print("=" * 70)
    print("  (the publish-date field mixes formats and cannot represent pre-1927 books; on")
    print("   the 1995+ subset the date is unambiguous and cover eras are close together, so")
    print("   any residual era confounding is minimal. Does the A-vs-C picture hold here?)")
    sub = frame[frame["book_age"] <= 26]
    sub_split = split.reindex(sub.index)
    train, test, controls = prep_train_test(sub, sub_split)
    for y_col in OUTCOMES:
        dr2, resA, resC, pA, pC = _delta_r2(train, test, y_col, controls)
        y = test[y_col].to_numpy(float)
        paired = _clustered_paired_test(y - pA, y - pC, test["primary_author"].to_numpy())
        tag = "reach " if y_col == OUTCOMES[0] else "rating"
        sg = resC.params["sal_gini"]
        lg = resC.params["legib_box_retention"]
        print(f"  [{tag}] train {len(train):5d} / test {len(test):4d}  heldout dR2={dr2:+.5f} "
              f"paired p={paired['p']:.3g}   sal_gini={sg:+.4f}(p={resC.pvalues['sal_gini']:.2g}) "
              f"legib={lg:+.4f}(p={resC.pvalues['legib_box_retention']:.2g})")
    print()


GEN_JSON = "data/generalization_results.json"


def main():
    frame = pd.read_parquet(FRAME)
    split = pd.read_parquet(SPLIT)["split"].reindex(frame.index)
    check_low_fame(frame, split)
    kfold = check_kfold(frame, split)
    check_nonlinearity(frame, split)
    check_modern_subset(frame, split)
    with open(GEN_JSON, "w") as f:
        json.dump({"kfold_delta_r2": kfold}, f, indent=2)
    print(f"wrote {GEN_JSON}")


if __name__ == "__main__":
    main()
