"""Deterministic author-grouped train/test split for the November benchmark.

GroupShuffleSplit on `primary_author` (seed 42, 80/20): every book by a given author lands
entirely in train or entirely in test, never both. This is the split the A-vs-C verdict
rests on -- it tests whether the cover features generalize to unseen authors rather than
letting a model (Model D especially) be rewarded for learning an author's visual style.
See build-log.md, November section, for why grouping beats a random split here.

Writes data/model_split.parquet (`key` -> "train" | "test") so every model uses the identical
partition. Re-run only if the working set changes.

  python -m scripts.modeling.split
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

FRAME = "data/model_frame.parquet"
OUT = "data/model_split.parquet"
SEED = 42
TEST_SIZE = 0.20


def make_split(df: pd.DataFrame) -> pd.Series:
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(gss.split(df, groups=df["primary_author"]))
    split = pd.Series("train", index=df.index, name="split")
    split.iloc[test_idx] = "test"
    return split


def main() -> None:
    df = pd.read_parquet(FRAME)
    split = make_split(df)
    split.to_frame().to_parquet(OUT)

    tr, te = df[split == "train"], df[split == "test"]
    a_tr, a_te = set(tr["primary_author"]), set(te["primary_author"])
    print(f"split -> {OUT}")
    print(f"  train {len(tr):,} rows / {len(a_tr):,} authors")
    print(f"  test  {len(te):,} rows / {len(a_te):,} authors")
    print(f"  author overlap train n test: {len(a_tr & a_te)}  (must be 0)")
    print()
    print("  outcome balance (train vs test):")
    for y in ("y_log_ratings", "y_rating"):
        print(f"    {y:16s} train mean={tr[y].mean():.3f} sd={tr[y].std():.3f}  "
              f"test mean={te[y].mean():.3f} sd={te[y].std():.3f}")
    print()
    gcols = [c for c in df.columns if c.startswith("g_")]
    drift = (te[gcols].mean() - tr[gcols].mean()).abs().sort_values(ascending=False)
    print("  largest genre-prevalence drift (test - train), top 6:")
    for c, d in drift.head(6).items():
        print(f"    {c[2:]:24s} {d:+.4f}  (train {tr[c].mean():.3f} / test {te[c].mean():.3f})")
    print(f"  mean |drift| across {len(gcols)} genres: {drift.mean():.4f}")


if __name__ == "__main__":
    main()
