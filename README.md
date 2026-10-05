# Do book-cover design rules predict book success?

> **Verdict:** No. Adding the five rules to a model that already knows genre, author fame, book age,
> series and publisher raises held-out R-squared on rating count by 0.06 percentage points on average
> across five author-grouped splits (pre-registered bar: 1 point), and does not improve average rating.
> Two rules, focal point and thumbnail legibility, point the opposite way from the advice.

Self-publishing and author-services blogs give writers confident, specific cover rules —
single focal point, high text/background contrast, two-font maximum, thumbnail legibility,
low clutter. A leading professional book designer (Peter Mendelsund, former Associate Art
Director at Knopf) says the opposite on record: design "should concern itself with
manufacturing desire," and "there is no science to this" — not a claim that rules are
worthless, but that no formula predicts success. This repo tests which side is closer to
right, on ~42k English-language books (from a ~52k Goodreads set), controlling for genre,
author fame, and other non-cover factors (book age, series, publisher).

Two articles came out of one build:

| # | Article | Month | What it covers |
|---|---|---|---|
| 1 | Architecture Teardown | September 2026 | Building the cover feature-extraction pipeline: [Can "Good Book Cover Design" Be Measured?](https://medium.com/@patrickcher/can-good-book-cover-design-be-measured-39ff30a38ff9) |
| 2 | Empirical Benchmark | October 2026 | Whether any of the 5 rules predict success once confounds are controlled: _link added on publication_ |

---

## Results

n = 42,344 English-language books. The test set is 8,499 books by 3,671 authors not seen in training
(split by author, seed 42). Reach is log10(rating count); reception is average rating.

**Held-out R-squared**

| Outcome | A: baseline | B: cover only | C: A + 5 rules | D: A + CLIP |
|---|--:|--:|--:|--:|
| Reach | 0.7464 | 0.0285 | 0.7472 | 0.7535 |
| Reception | 0.3641 | 0.0004 | 0.3628 | 0.3698 |

**Does adding the five rules help? (C minus A, held-out R-squared)**

| Outcome | Single split | 95% CI | 5-fold mean | Clears the 0.010 bar? |
|---|--:|---|--:|:-:|
| Reach | +0.0008 | [+0.0001, +0.0016] | +0.0006 (one fold negative) | no |
| Reception | -0.0013 | [-0.0022, -0.0005] | +0.00001 (sign unstable) | no |

**Per-rule effects in Model C, per standard deviation of the feature**

| Rule | Effect on reach | Effect on rating (stars) |
|---|--:|--:|
| Single focal point | -4.6% | +0.0014 |
| Text/background contrast | +2.5% | +0.0030 |
| Two-font maximum (text-block count as a stand-in) | +1.4% (p = 0.054) | +0.0002 |
| Thumbnail legibility | -1.5% | -0.0059 |
| Low clutter | -2.0% (less clutter helps) | +0.0009 |

Across ten tests (five rules, two outcomes), three survive a Bonferroni correction: focal point on reach
and thumbnail legibility on rating (both against the advice), and contrast on reach (with it, but small).
Rendered tables are in [`article-assets/model-comparison.md`](article-assets/model-comparison.md); the raw
numbers are in `data/model_results.json`, `data/model_comparison.csv`, `data/model_c_coefficients.csv`,
`data/within_author_coefficients.csv` and `data/generalization_results.json`.

See [`docs/modeling.md`](docs/modeling.md) for the four-model method (baseline, cover-only,
rules, CLIP), the robustness checks and how to run it yourself.

**Data.** [Best Books Ever dataset](https://zenodo.org/records/4265096) (Costa Planells and Casanova
Lozano, Goodreads, CC BY-NC 4.0).

---

## Running it yourself

Everything is reproducible from a clean checkout. The how-to lives in `docs/`:

- [`docs/setup.md`](docs/setup.md) — environment (there's a non-obvious `visual-clutter` install)
- [`docs/data.md`](docs/data.md) — getting and validating the dataset
- [`docs/pipeline.md`](docs/pipeline.md) — the 5 features + CLIP, running extraction (smoke and full)
- [`docs/outputs.md`](docs/outputs.md) — every file the analysis consumes, with schema
- [`docs/modeling.md`](docs/modeling.md) — the benchmark: the four models, the split, the pre-registered bar, how to run it
- [`notebooks/`](notebooks/) — exploratory analysis (not on the reproducibility path)

Some docs and code comments cite `build-log.md`, the author's working log. It is not part of this repo.
