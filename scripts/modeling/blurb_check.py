"""Blurb baseline: does the cover add anything beyond what the book's own description says?

Model A controls for genre, author fame, book age, series, publisher. This check adds a
compressed representation of the book's back-cover blurb (`description`) to that baseline and
asks the A-vs-C question again: with the blurb also held fixed, do the five cover rules add
anything to predicting reach or reception?

Blurb features: TF-IDF (fit on train only) -> TruncatedSVD to 80 components. Missing blurb
(~1%) -> zeros + a flag.

  python -m scripts.modeling.blurb_check        # -> stdout + build-log

Reads data/model_frame.parquet, data/model_split.parquet, data/bbe_github.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.modeling.models import (
    FAME_BIN,
    FAME_CONT,
    FEATURE_BIN,
    FEATURE_CONT,
    SEED,
    _design,
    _fit,
    _r2,
    prep_train_test,
)

FRAME = "data/model_frame.parquet"
SPLIT = "data/model_split.parquet"
RAW = "data/bbe_github.csv"
N_SVD = 80
OUTCOMES = {"y_log_ratings": "reach ", "y_rating": "rating"}


def blurb_features(index: pd.Index, is_train: np.ndarray) -> pd.DataFrame:
    raw = pd.read_csv(RAW, dtype=str)
    raw = raw[~raw["bookId"].duplicated(keep="first")].set_index("bookId").reindex(index)
    desc = raw["description"].fillna("")
    missing = (desc.str.strip() == "").to_numpy().astype(int)

    tfidf = TfidfVectorizer(max_features=15000, min_df=5, stop_words="english", sublinear_tf=True)
    tfidf.fit(desc[is_train])
    X = tfidf.transform(desc)
    svd = TruncatedSVD(n_components=N_SVD, random_state=SEED)
    svd.fit(X[is_train])
    pcs = svd.transform(X)
    # standardize on train
    mu, sd = pcs[is_train].mean(0), pcs[is_train].std(0)
    sd[sd == 0] = 1.0
    pcs = (pcs - mu) / sd
    cols = [f"blurb_pc{i:02d}" for i in range(N_SVD)]
    out = pd.DataFrame(pcs, index=index, columns=cols)
    out["blurb_missing"] = missing
    out.attrs["evr"] = float(svd.explained_variance_ratio_.sum())
    return out


def main() -> None:
    frame = pd.read_parquet(FRAME)
    split = pd.read_parquet(SPLIT)["split"].reindex(frame.index)
    train, test, ctrl = prep_train_test(frame, split)
    is_train = (split == "train").to_numpy()

    blurb = blurb_features(frame.index, is_train)
    bcols = list(blurb.columns)
    train = train.join(blurb)
    test = test.join(blurb)
    print(f"blurb: {N_SVD} SVD components retain {blurb.attrs['evr']:.1%} of TF-IDF variance "
          f"({blurb['blurb_missing'].mean():.1%} of books have no blurb)\n")

    base = FAME_CONT + FAME_BIN + ctrl
    feats = FEATURE_CONT + FEATURE_BIN
    specs = {
        "A            ": base,
        "A+blurb      ": base + bcols,
        "C  (A+covers)": base + feats,
        "C+blurb      ": base + bcols + feats,
    }
    for y_col, tag in OUTCOMES.items():
        r2 = {}
        for name, cols in specs.items():
            res = _fit(_design(train, cols), train[y_col])
            r2[name.strip()] = _r2(test[y_col].to_numpy(float),
                                   np.asarray(res.predict(_design(test, cols)), float))
        print(f"[{tag}] held-out R2:")
        for name in specs:
            print(f"   {name}  {r2[name.strip()]:.4f}")
        print(f"   blurb adds over A            : {r2['A+blurb'] - r2['A']:+.5f}")
        print(f"   covers add over A            : {r2['C  (A+covers)'] - r2['A']:+.5f}")
        print(f"   covers add over A+blurb      : {r2['C+blurb'] - r2['A+blurb']:+.5f}   <- the test")
        print()


if __name__ == "__main__":
    main()
