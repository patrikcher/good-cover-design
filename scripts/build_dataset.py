"""Build the canonical working set from the raw CSV.

Decisions (see docs/data.md):
  * English only -- the OCR features use an English recognizer; non-English covers would
    inject noise into 3 of the 5 features.
  * Drop rows with numRatings == 0 (no outcome signal) and non-http coverImg.
  * De-dup on bookId (keep first).
  * Keep rows with empty genres -- flagged, not dropped; the Nov baseline model handles it.

  python -m scripts.build_dataset

Output: data/books_english.parquet  (index = key = bookId)
"""
from __future__ import annotations

import ast
import re

import pandas as pd

from scripts.cover_io import canonical_url

RAW = "data/bbe_github.csv"
OUT = "data/books_english.parquet"

_PARENS = re.compile(r"\s*\([^)]*\)")


def primary_author(a: str) -> str:
    """First credited person, annotations removed. 'J.K. Rowling, Mary GrandPré (Illustrator)'
    -> 'J.K. Rowling'.  'Nathan Erez (Goodreads Author)' -> 'Nathan Erez'."""
    a = str(a).split(",")[0]
    a = _PARENS.sub("", a)
    return re.sub(r"\s+", " ", a).strip(" .").strip() or "Unknown"


def parse_genres(g: str) -> list[str]:
    try:
        v = ast.literal_eval(g)
        return [str(x) for x in v] if isinstance(v, list) else []
    except (ValueError, SyntaxError):
        return []


def main() -> None:
    df = pd.read_csv(RAW, low_memory=False)
    n0 = len(df)

    df = df[df["language"] == "English"].copy()
    n_en = len(df)

    df = df[df["numRatings"] > 0]
    df = df[df["coverImg"].astype(str).str.startswith("http")]
    df = df.drop_duplicates(subset="bookId", keep="first")

    df["key"] = df["bookId"]
    df["primary_author"] = df["author"].map(primary_author)
    df["genres_list"] = df["genres"].map(parse_genres)
    df["primary_genre"] = df["genres_list"].map(lambda xs: xs[0] if xs else None)
    df["has_genre"] = df["genres_list"].map(len) > 0
    df["cover_url"] = df["coverImg"].map(canonical_url)

    keep = [
        "key", "title", "author", "primary_author",
        "genres_list", "primary_genre", "has_genre",
        "rating", "numRatings", "bbeScore", "bbeVotes",
        "cover_url",
    ]
    out = df[keep].set_index("key")
    out.to_parquet(OUT)

    print(f"raw rows                : {n0}")
    print(f"English                 : {n_en}")
    print(f"after numRatings>0 / cover / dedup : {len(out)}")
    print(f"  with non-empty genres : {out['has_genre'].sum()} ({100*out['has_genre'].mean():.1f}%)")
    print(f"  distinct primary authors : {out['primary_author'].nunique()}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
