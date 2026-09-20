"""Load-bearing check: what would have gone wrong WITHOUT leave-one-out author fame.

build-log.md / docs/outputs.md assert that a naive fame proxy (an author's total rating
count across the dataset, including the book itself) would let a book's own success leak
into its own confound covariate. This reproduces the naive version and shows the leakage
concretely, rather than just asserting the fix was necessary.

  python -m scripts.analysis.author_fame_leakage
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

books = pd.read_parquet("data/books_english.parquet")
fame = pd.read_parquet("data/author_fame.parquet")
df = books.join(fame)

g = df.groupby("primary_author")
naive_sum = g["numRatings"].transform("sum")           # includes this book
naive_mean_rating = g["rating"].transform("mean")       # includes this book

n = len(df)
print(f"n = {n} books, {df['primary_author'].nunique()} authors\n")

print("spearman(fame proxy, its own outcome) -- naive (leaky) vs leave-one-out:\n")
r_naive_n = spearmanr(naive_sum, df["numRatings"]).statistic
r_loo_n = spearmanr(df["author_sum_ratings_loo"], df["numRatings"]).statistic
print(f"  sum_ratings  vs numRatings :  naive {r_naive_n:+.3f}   ->   LOO {r_loo_n:+.3f}")

m = df["author_mean_rating_loo"].notna()
r_naive_r = spearmanr(naive_mean_rating[m], df.loc[m, "rating"]).statistic
r_loo_r = spearmanr(df.loc[m, "author_mean_rating_loo"], df.loc[m, "rating"]).statistic
print(f"  mean_rating  vs rating    :  naive {r_naive_r:+.3f}   ->   LOO {r_loo_r:+.3f}")

print()
solo = df["author_is_solo"]
print(f"the starkest case -- solo authors ({solo.sum()} books, {100*solo.mean():.1f}% of the set):")
print(f"  for a solo author, naive sum_ratings IS this book's numRatings, exactly.")
print(f"  spearman(naive_sum, numRatings) among solo authors = "
      f"{spearmanr(naive_sum[solo], df.loc[solo, 'numRatings']).statistic:+.4f}  (perfect, by construction)")
loo_solo = df.loc[solo, "author_sum_ratings_loo"]
assert (loo_solo == 0).all(), "expected LOO sum to be exactly 0 for every solo author"
print(f"  LOO_sum among solo authors = 0 for all {solo.sum()} of them, exactly, by construction "
      f"(no other book to sum) -- correlation is undefined on a constant, which is the point: "
      f"LOO correctly contributes zero fame signal for these books instead of smuggling in the answer.")

print()
print("what this means for Model A: a naive fame covariate would have let the model partly")
print("'predict' numRatings using numRatings itself for the 28.7% of books whose author has")
print("no other book in the set -- inflating Model A's apparent explanatory power, which would")
print("then understate how much Model C (cover features) adds on top of it.")
