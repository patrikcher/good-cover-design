# Data

## Source: use the GitHub CSV, not the Kaggle mirror

The widely-cited Kaggle mirror `arnabchaki/goodreads-best-books-ever` **won't work here.**
It ships 23 columns, not 25 — it dropped `bookId` **and `coverImg`** (the cover URL, which the
whole pipeline needs) and mangled `isbn` into Excel scientific notation (`9.78044E+12`).

Use the upstream source instead:

```bash
mkdir -p data
curl -sL -o data/bbe_github.csv \
  "https://raw.githubusercontent.com/scostap/goodreads_bbe_dataset/main/Best_Books_Ever_dataset/books_1.Best_Books_Ever.csv"
```

52,478 rows × 25 columns. This file is the dataset of record. It's gitignored (70 MB).

## Field completeness (validated, not trusted)

`python -m scripts.validate_dataset` re-checks this on the actual file:

| Field | Completeness | Notes |
|---|---|---|
| `title`, `author` | 100% | |
| `rating` (avg), `numRatings` | 99.9% | 71 rows are genuinely 0 on both — drop them |
| `genres` | 91.2% usable | 4,623 rows are literally `"[]"` |
| `coverImg` | 98.85% | 605 blank; single host `i.gr-assets.com`; no `nophoto` placeholders |
| `isbn` | 91.7% real | 4,354 (8.3%) are the placeholder `9999999999999` — don't rely on ISBN for cover fallback |
| `language` | 81.3% English | 42,661 English; rest is 60+ languages |

Other cleanup flags: 54 duplicate `bookId`, 88 duplicate `(title, author)`.

## Cover images

Covers are fetched from the `coverImg` URL. Two things to know:

1. **Strip the downscale suffix.** ~20% of URLs carry `._SY475_`, `._SX318_`, or the compound
   `._SX318_SY475_` before the extension. `cover_io.canonical_url` removes it
   (`\._(?:S[XY]\d+_)+(?=\.\w+$)`) — for ~20% of covers this returns a larger master, for the
   rest a byte-identical file. A naive regex that misses the compound form produces a 403 URL,
   so use the one in `cover_io`.
2. **Everything is resized to 480 px tall.** Goodreads serves compressed thumbnails (~314×475
   median; some masters go to 2560 px after suffix-stripping, most don't). `cover_io.load_cover`
   normalizes every cover to 480 px height so source resolution is never a confound in the
   features. This is a ~480 px corpus — treat it as one.

Consequence for the **thumbnail-legibility** feature: its 160 px downscale is only ~3× from the
480 px base, gentler than a true print→browse-thumbnail cliff. If that feature comes out null
in the modeling, "test too gentle to detect it" is a live explanation.

## Sample for smoke tests

```bash
python -m scripts.fetch_sample            # 50 covers, seed 42 -> data/sample_covers/
```

Deterministic given `(n, seed)` — the same books every run, so the smoke-test numbers in
`build-log.md` reproduce.
