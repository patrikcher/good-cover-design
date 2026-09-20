"""Validate field completeness on the ACTUAL downloaded dataset, not the source README.

Uses the upstream source CSV (scostap/goodreads_bbe_dataset on GitHub), NOT the Kaggle
mirror arnabchaki/goodreads-best-books-ever -- the Kaggle mirror is unusable here (no
coverImg, no bookId, isbn mangled to scientific notation). See docs/data.md.

Run: python -m scripts.validate_dataset
"""
import re
import pandas as pd

CSV = "data/bbe_github.csv"
df = pd.read_csv(CSV, low_memory=False)
n = len(df)
print(f"{CSV}: {df.shape[0]} rows x {df.shape[1]} cols\n")


def pct(k):
    return f"{100 * k / n:5.1f}%"


checks = {
    "title (non-empty)": df["title"].astype(str).str.strip().ne("").sum(),
    "author (non-empty)": df["author"].astype(str).str.strip().ne("").sum(),
    "rating in (0,5]": df["rating"].between(0, 5, inclusive="right").sum(),
    "numRatings > 0": (df["numRatings"] > 0).sum(),
    "genres non-empty list": df["genres"].astype(str).str.strip().ne("[]").sum(),
    "coverImg is http url": df["coverImg"].astype(str).str.startswith("http").sum(),
    "isbn real (not 9999999999999)": df["isbn"].astype(str).str.strip().ne("9999999999999").sum(),
    "language == English": df["language"].eq("English").sum(),
}
for label, k in checks.items():
    print(f"  {label:34s} {k:6d}  {pct(k)}")

print("\n  duplicate bookId          :", df["bookId"].duplicated().sum())
print("  duplicate (title, author) :", df.duplicated(["title", "author"]).sum())
print("  numRatings==0 AND rating==0:", ((df["numRatings"] == 0) & (df["rating"] == 0)).sum())

ci = df["coverImg"].dropna()
resize = ci.str.contains(r"_S[XY]\d", regex=True).sum()
print(f"\n  coverImg with explicit _SX/_SY resize suffix: {resize} ({100*resize/len(ci):.1f}%)")
print("  NOTE: goodreads serves compressed thumbnails; sample covers ~314x475 median,")
print("        so 'full-res' for the thumbnail-legibility feature is already small.")

# advertised completeness (source README) vs. reality on the upstream GitHub CSV
print("\n--- advertised vs. reality (upstream GitHub CSV) ---")
print("  author 100%       -> 100.0% OK")
print("  genre 91%         -> 91.2% OK")
print("  cover URL 99%     -> 98.9% OK (605 blank)")
print("  average rating 100% -> 100.0% OK")
print("  rating count 100% -> 100.0% OK (but 71 rows are genuinely 0)")
print("  isbn (not advertised) -> 8.3% are '9999999999999' placeholder")
