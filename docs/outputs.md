# Pipeline outputs (inputs to the modeling)

All keyed by `key` = the dataset's `bookId` slug (e.g. `2767052-the-hunger-games`; some run to
~130 chars). Join on it.

## `data/books_english.parquet` — the working set

`python -m scripts.build_dataset`. 42,345 rows (English only; `numRatings > 0`; has an http
cover; deduped on bookId).

| Column | Meaning |
|---|---|
| `title`, `author` | raw strings |
| `primary_author` | first credited person, annotations stripped — the key for author-fame |
| `genres_list` | list[str], full genre tags (may be empty) |
| `primary_genre` | first tag, or None (6.1% have no genre) |
| `has_genre` | bool |
| `rating` | Goodreads average rating — **outcome 2** |
| `numRatings` | rating count — **outcome 1** |
| `bbeScore`, `bbeVotes` | Goodreads "Best Books Ever" community vote — context only, not an outcome (survivorship-biased, circular with author fame) |
| `cover_url` | canonicalized cover URL |

## `data/author_fame.parquet` — confound covariates

`python -m scripts.author_fame`. All **leave-one-out** (computed from the author's *other*
books) so a book's own success never leaks into its fame covariate.

| Column | Meaning |
|---|---|
| `author_n_books_loo` | # of the author's other books in the set (0–90; median 3, mean 7.9) |
| `author_sum_ratings_loo` | total `numRatings` across the author's other books |
| `author_max_ratings_loo` | largest single `numRatings` among the author's other books ("has a hit") |
| `author_mean_rating_loo` | mean `rating` across the author's other books (NaN if solo) |
| `author_is_solo` | True → author has no other book here; all LOO sums are 0 |

~28.7% of books are by a solo author. Give the models `author_is_solo` alongside the LOO
values so "no fame data" is distinguishable from "fame = 0".

## `data/features_full.parquet` — the 5 interpretable features

`python -m scripts.extract_features --books --out data/features_full.parquet`. Schema and
per-feature caveats: [`pipeline.md`](pipeline.md). NaNs where OCR found no text (~8%) — carry a
missingness indicator into the models, don't impute silently.

## `data/clip_embeddings.npz` — the catch-all (Model D only)

`python -m scripts.extract_clip`. `np.load(path, allow_pickle=True)` →

| Array | Shape / dtype | |
|---|---|---|
| `keys` | (N,) object | bookId slugs |
| `embeddings` | (N, 512) float32 | open_clip ViT-B-32 / laion2b, L2-normalized |

Model D is Model C's metadata + these 512 dims (reduced — PCA/PLS — before fitting). It's an
upper-bound diagnostic on "signal in the cover", not an interpretable result. Compare it to
Model C, not to the baseline.

## Coverage / join notes

- Not every working-set row will have a feature row or an embedding: some covers 404 or fail
  to decode. Expect ~1–2% shortfall. Left-join on `books_english` and treat missing feature
  rows explicitly.
- `data/covers/<key>.jpg` is the local cover store (gitignored, ~2 GB for the full set).
