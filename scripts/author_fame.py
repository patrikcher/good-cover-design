"""Derive author-fame proxies from the dataset itself (no external 'fame' field exists).

Confound control for the November models: a famous author both sells more and can afford a
better cover, so fame has to be conditioned on.

All proxies are **leave-one-out** -- computed from the author's *other* books -- so the book's
own success never leaks into its own fame covariate.

  python -m scripts.author_fame

Output: data/author_fame.parquet  (index = key = bookId)
Columns:
  author_n_books_loo      : # of the author's OTHER books in the dataset
  author_sum_ratings_loo  : total numRatings across the author's OTHER books
  author_max_ratings_loo  : largest single numRatings among the author's OTHER books ("has a hit")
  author_mean_rating_loo  : mean of `rating` across the author's OTHER books
  author_is_solo          : True if the author has no other book here (all LOO sums are 0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SRC = "data/books_english.parquet"
OUT = "data/author_fame.parquet"


def main() -> None:
    df = pd.read_parquet(SRC)[["primary_author", "numRatings", "rating"]].copy()

    g = df.groupby("primary_author")
    grp_n = g["numRatings"].transform("size")
    grp_sum = g["numRatings"].transform("sum")
    grp_rsum = g["rating"].transform("sum")
    grp_max = g["numRatings"].transform("max")
    grp_max_count = g["numRatings"].transform(lambda s: int((s == s.max()).sum()))
    grp_2nd = g["numRatings"].transform(
        lambda s: (np.sort(s.unique())[-2] if s.nunique() >= 2 else 0)
    )

    n_loo = (grp_n - 1).astype(int)
    # max among the author's OTHER books: 2nd-distinct value only if this row is the sole max
    max_loo = np.where((df["numRatings"] == grp_max) & (grp_max_count == 1), grp_2nd, grp_max)
    max_loo = np.where(grp_n == 1, 0, max_loo)

    out = pd.DataFrame(index=df.index)
    out["author_n_books_loo"] = n_loo
    out["author_sum_ratings_loo"] = (grp_sum - df["numRatings"]).astype(int)
    out["author_max_ratings_loo"] = max_loo.astype("int64")
    out["author_mean_rating_loo"] = np.where(
        n_loo > 0, (grp_rsum - df["rating"]) / n_loo.replace(0, np.nan), np.nan
    )
    out["author_is_solo"] = n_loo == 0

    out.to_parquet(OUT)
    print(f"{len(out)} books, {df['primary_author'].nunique()} authors")
    print(f"  books by a solo author (no other book here): {out['author_is_solo'].sum()} "
          f"({100*out['author_is_solo'].mean():.1f}%)")
    print(out.drop(columns="author_is_solo").describe(percentiles=[.5, .9, .99]).T.round(1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
