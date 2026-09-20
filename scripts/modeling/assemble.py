"""Assemble the single modeling frame for the November benchmark.

Joins the pipeline outputs into one table, one row per cover:
  - two outcomes, kept separate: log10 rating count, average rating
  - control covariates: author-fame LOO proxies + genre multi-hot
  - the 5 interpretable claim features, each with an explicit missingness flag
  - primary_author, for the author-grouped train/test split

CLIP embeddings are NOT joined here -- they stay in data/clip_embeddings.npz and are aligned
by `key` only in the Model D step.

  python -m scripts.modeling.assemble        # -> data/model_frame.parquet + a summary

Feature provenance and per-feature caveats: docs/pipeline.md. Methodology decisions (split,
model family): build-log.md, November section.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

BOOKS = "data/books_english.parquet"
FAME = "data/author_fame.parquet"
FEATURES = "data/features_full.parquet"
RAW = "data/bbe_github.csv"
OUT = "data/model_frame.parquet"

# A genre becomes a control dummy if it tags at least this fraction of the working set.
GENRE_MIN_FRAC = 0.01
# A publisher becomes a control dummy if it appears on at least this fraction of the set.
PUBLISHER_MIN_FRAC = 0.003
# The dataset was scraped in 2021; book age is measured back from here.
REF_YEAR = 2021
SELF_PUB = {
    "createspace independent publishing platform", "createspace", "independently published",
    "amazon digital services", "amazon digital services llc", "smashwords",
    "smashwords edition", "lulu.com", "lulu", "kindle edition", "kindle direct publishing",
    "kindle", "self-published", "self published", "xlibris", "authorhouse", "iuniverse",
}

# claim -> the primary feature column(s). Collinear twins (sal_top10_mass, median contrast,
# legib_char_retention, the _sr saliency cols) are left in features_full.parquet for
# robustness checks and deliberately not carried here. See docs/pipeline.md.
CLAIM_FEATURES = {
    "single_focal_point": ["sal_gini"],
    "text_bg_contrast": ["title_contrast_ratio"],
    "typographic_simplicity": ["text_block_count"],
    "thumbnail_legibility": ["legib_box_retention"],
    "low_clutter": ["feature_congestion", "subband_entropy"],
}
FEATURE_COLS = [c for cols in CLAIM_FEATURES.values() for c in cols]
# features that carry NaNs (wordless covers / detector found nothing) -> flag + median-impute
IMPUTE_COLS = ["title_contrast_ratio", "legib_box_retention"]


def _slug(name: str) -> str:
    return "g_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _pub_slug(name: str) -> str:
    return "pub_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:30]


_YEAR4 = re.compile(r"\b(1[5-9]\d\d|20[0-2]\d)\b")


def _parse_year(s):
    """Publication date -> 4-digit year. The field mixes 'MM/DD/YY' and 'Month Nth YYYY'
    and bare 'YYYY'. Slash form pivots at 26 (<=26 -> 20xx, else 19xx), so pre-1927 books
    read as recent; a suspect flag catches the obvious cases (Classics dated after 1970)
    and the post-1995 subset check backstops the rest. See build-log.md."""
    if not isinstance(s, str):
        return np.nan
    p = s.split("/")
    if len(p) == 3 and p[2].isdigit() and len(p[2]) == 2:
        yy = int(p[2])
        return float(2000 + yy if yy <= 26 else 1900 + yy)
    hit = _YEAR4.search(s)
    return float(hit.group(1)) if hit else np.nan


def _meta_controls(index: pd.Index, primary_genre: pd.Series) -> pd.DataFrame:
    """Publication era, series, and publisher, joined from the raw CSV on bookId (== key)."""
    raw = pd.read_csv(RAW, dtype=str)
    raw = raw[~raw["bookId"].duplicated(keep="first")].set_index("bookId").reindex(index)
    m = pd.DataFrame(index=index)

    # --- publication era ---
    yr = raw["firstPublishDate"].map(_parse_year).fillna(raw["publishDate"].map(_parse_year))
    m["pub_year_missing"] = yr.isna().astype(int)
    yr = yr.fillna(yr.median()).clip(1900, REF_YEAR)
    m["book_age"] = (REF_YEAR - yr).astype(float)
    # pre-1927 classics misparse as recent; flag the obvious ones so the model can absorb them
    m["pub_year_suspect"] = (
        primary_genre.str.lower().eq("classics").fillna(False) & yr.gt(1970)
    ).astype(int)

    # --- series ---
    series = raw["series"].fillna("").str.replace(r"\s*#\s*[\d.,\- ]+$", "", regex=True).str.strip()
    m["has_series"] = (series != "").astype(int)
    size = series.replace("", np.nan).map(series[series != ""].value_counts()).fillna(0)
    m["log_series_size"] = np.log1p(size.to_numpy(dtype=float))

    # --- publisher ---
    pub = raw["publisher"].fillna("").str.strip()
    pub_lower = pub.str.lower()
    m["publisher_missing"] = (pub == "").astype(int)
    m["self_published"] = pub_lower.isin(SELF_PUB).astype(int)
    freq = pub[(pub != "") & (m["self_published"] == 0)].value_counts()
    keep_pub = freq[freq >= PUBLISHER_MIN_FRAC * len(index)].index.tolist()
    pub_cols = [_pub_slug(p) for p in keep_pub]
    dummies = pd.DataFrame(
        {_pub_slug(p): (pub == p).astype(int) for p in keep_pub}, index=index
    )
    m = pd.concat([m, dummies], axis=1)
    m.attrs["pub_cols"] = pub_cols
    return m


def build_frame() -> pd.DataFrame:
    books = pd.read_parquet(BOOKS)
    fame = pd.read_parquet(FAME)
    feats = pd.read_parquet(FEATURES)[FEATURE_COLS]

    n_books = len(books)
    df = books.join(feats, how="inner").join(fame, how="left")
    dropped_no_feat = n_books - len(df)
    missing_fame = df["author_n_books_loo"].isna().sum()
    if missing_fame:
        raise ValueError(f"{missing_fame} rows have no author-fame row; expected exact coverage")

    out = pd.DataFrame(index=df.index)
    out.index.name = "key"
    out["primary_author"] = df["primary_author"]

    # --- outcomes (kept separate, never blended) ---
    out["y_log_ratings"] = np.log10(df["numRatings"].to_numpy(dtype=float))  # numRatings >= 1
    out["y_rating"] = df["rating"].astype(float)
    out["numRatings"] = df["numRatings"]  # raw, for reference / count-GLM robustness check
    out["rating"] = df["rating"]

    # --- control: author fame (all LOO, so a book's own success never leaks in) ---
    out["fame_log_sum_ratings"] = np.log1p(df["author_sum_ratings_loo"].to_numpy(dtype=float))
    out["fame_log_max_ratings"] = np.log1p(df["author_max_ratings_loo"].to_numpy(dtype=float))
    out["fame_log_n_books"] = np.log1p(df["author_n_books_loo"].to_numpy(dtype=float))
    mean_rating = df["author_mean_rating_loo"].astype(float)
    out["fame_mean_rating_missing"] = mean_rating.isna().astype(int)
    out["fame_mean_rating"] = mean_rating.fillna(mean_rating.median())
    out["author_is_solo"] = df["author_is_solo"].astype(int)

    # --- control: genre multi-hot (>= GENRE_MIN_FRAC of the working set) ---
    # These are Goodreads *shelf* tags, not a curated taxonomy -- the set includes some
    # format/meta tags (audiobook, novels, unfinished). Kept as-is: a single frequency
    # threshold with no hand-curation, used only as controls to make Model C's test harder.
    genres = df["genres_list"].apply(lambda g: list(g) if g is not None else [])
    freq = genres.explode().value_counts()
    keep = freq[freq >= GENRE_MIN_FRAC * len(df)].index.tolist()
    keep_set = set(keep)
    genre_cols = [_slug(name) for name in keep]
    genre_dummies = pd.DataFrame(
        {_slug(name): genres.apply(lambda gs, k=name: int(k in gs)) for name in keep},
        index=out.index,
    )
    out = pd.concat([out, genre_dummies], axis=1)
    out["has_genre"] = df["has_genre"].astype(int)
    out["n_genre_tags_kept"] = genres.apply(lambda gs: sum(g in keep_set for g in gs))

    # --- control: publication era, series, publisher (joined from the raw CSV) ---
    meta = _meta_controls(out.index, df["primary_genre"])
    pub_cols = meta.attrs["pub_cols"]
    meta_cont = ["book_age", "log_series_size"]
    meta_bin = ["pub_year_missing", "pub_year_suspect", "has_series", "self_published",
                "publisher_missing"]
    out = pd.concat([out, meta], axis=1)

    # --- the 5 claim features, with missingness flags ---
    for c in FEATURE_COLS:
        vals = df[c].astype(float)
        if c in IMPUTE_COLS:
            out[f"{c}_missing"] = vals.isna().astype(int)
            vals = vals.fillna(vals.median())
        out[c] = vals

    modeling_cols = (
        ["y_log_ratings", "y_rating"]
        + [c for c in out.columns if c.startswith("fame_")]
        + ["author_is_solo", "has_genre"]
        + genre_cols
        + meta_cont + meta_bin + pub_cols
        + FEATURE_COLS
        + [f"{c}_missing" for c in IMPUTE_COLS]
    )
    bad = out[modeling_cols].isna().sum()
    if bad.any():
        raise ValueError(f"NaNs left in modeling columns:\n{bad[bad > 0]}")

    out.attrs["genre_cols"] = genre_cols
    out.attrs["meta_cont"] = meta_cont
    out.attrs["meta_bin"] = meta_bin
    out.attrs["pub_cols"] = pub_cols
    out.attrs["feature_cols"] = FEATURE_COLS
    out.attrs["dropped_no_feature"] = dropped_no_feat
    return out


def main() -> None:
    out = build_frame()
    out.to_parquet(OUT)

    gcols = out.attrs["genre_cols"]
    print(f"model frame: {len(out):,} rows  ->  {OUT}")
    print(f"  dropped (no feature row): {out.attrs['dropped_no_feature']}")
    print(f"  genre dummies (>= {GENRE_MIN_FRAC:.0%} of rows): {len(gcols)}")
    print(f"  publisher dummies (>= {PUBLISHER_MIN_FRAC:.1%} of rows): {len(out.attrs['pub_cols'])}")
    print(f"    {', '.join(c[4:] for c in out.attrs['pub_cols'])}")
    print()
    print("  metadata controls:")
    age = out["book_age"]
    print(f"    book_age: p10={age.quantile(.1):.0f} p50={age.median():.0f} p90={age.quantile(.9):.0f} "
          f"(missing {out['pub_year_missing'].mean():.1%}, suspect {out['pub_year_suspect'].sum():,})")
    print(f"    has_series {out['has_series'].mean():.1%}   self_published {out['self_published'].mean():.1%}   "
          f"publisher_missing {out['publisher_missing'].mean():.1%}")
    print()
    print("  outcomes:")
    for y in ("y_log_ratings", "y_rating"):
        s = out[y]
        print(f"    {y:16s} mean={s.mean():.3f} sd={s.std():.3f} "
              f"min={s.min():.3f} p50={s.median():.3f} max={s.max():.3f}")
    print(f"  raw numRatings: p50={out['numRatings'].median():,.0f} "
          f"p90={out['numRatings'].quantile(.9):,.0f} max={out['numRatings'].max():,.0f}")
    print()
    print("  missingness flags:")
    for c in IMPUTE_COLS:
        f = out[f"{c}_missing"]
        print(f"    {c}_missing: {f.sum():,} ({f.mean():.1%})")
    print()
    print("  feature <-> outcome (spearman, univariate, no controls -- teaser only):")
    from scipy.stats import spearmanr

    for c in out.attrs["feature_cols"]:
        r1 = spearmanr(out[c], out["y_log_ratings"]).statistic
        r2 = spearmanr(out[c], out["y_rating"]).statistic
        print(f"    {c:24s} vs log_ratings {r1:+.3f}   vs rating {r2:+.3f}")
    print()
    grp = out["primary_author"].nunique()
    multi = (out["primary_author"].map(out["primary_author"].value_counts()) > 1).sum()
    print(f"  split groups: {grp:,} distinct authors; {multi:,} rows ({multi / len(out):.0%}) "
          f"share an author with another row")


if __name__ == "__main__":
    main()
