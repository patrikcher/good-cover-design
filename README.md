# Do book-cover design rules predict book success?

> **Verdict:** TBD — full write-up publishing November 2026.

Self-publishing and author-services blogs give writers confident, specific cover rules —
single focal point, high text/background contrast, two-font maximum, thumbnail legibility,
low clutter. A leading professional book designer (Peter Mendelsund, former Associate Art
Director at Knopf) says the opposite on record: "There is no science to this." This repo
tests which side is closer to right, on ~42k English-language books (from a ~52k Goodreads
set), controlling for genre, author fame, and other non-cover factors (book age, series,
publisher).

Two articles came out of one build:

| # | Article | Month | What it covers |
|---|---|---|---|
| 1 | Architecture Teardown | October 2026 | Building the cover feature-extraction pipeline — _link TBD_ |
| 2 | Empirical Benchmark | November 2026 | Whether any of the 5 rules predict success once confounds are controlled — _link TBD_ |

---

## Results

_Full results and analysis publish alongside Article 2 in November 2026._

See [`docs/modeling.md`](docs/modeling.md) for the four-model method (baseline, cover-only,
rules, CLIP) and how to run it yourself.

---

## Running it yourself

Everything is reproducible from a clean checkout. The how-to lives in `docs/`:

- [`docs/setup.md`](docs/setup.md) — environment (there's a non-obvious `visual-clutter` install)
- [`docs/data.md`](docs/data.md) — getting and validating the dataset
- [`docs/pipeline.md`](docs/pipeline.md) — the 5 features + CLIP, running extraction (smoke and full)
- [`docs/outputs.md`](docs/outputs.md) — every file the analysis consumes, with schema
- [`docs/modeling.md`](docs/modeling.md) — the November benchmark: the four models, the split, the pre-registered bar, how to run it
- [`notebooks/`](notebooks/) — exploratory analysis (not on the reproducibility path)
