# Do book-cover design rules predict book success?

> **Verdict:** On 42,344 English-language books, controlling for genre, author fame, book age,
> series, and publisher, the five rules add **no improvement that survives re-splitting the
> data** to predicting a book's reach (rating count), and **nothing** to its average rating.
> Two of the five individual rules — single focal point and thumbnail legibility — are
> statistically clear and point *against* the advice. A full CLIP image embedding (the larger
> ViT-L-14), the ceiling on any cover signal, adds ≈ 0.7% over the same baseline. Mendelsund's
> "there is no science to this" is much closer to right.
>
> _(Benchmark run complete; Article 2 write-up in progress.)_

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

Two outcomes, reported separately. Held-out = 8,499 books by 3,671 authors not seen in
training. Full tables: [`article-assets/model-comparison.md`](article-assets/model-comparison.md).

**Held-out R², four models** (A = genre + author fame + book age + series + publisher):

| Outcome | A: baseline | B: cover only | C: A + 5 rules | D: A + CLIP (ViT-L-14) |
|---|--:|--:|--:|--:|
| Reach — log10(rating count) | 0.746 | 0.028 | 0.747 | 0.754 |
| Reception — average rating | 0.364 | 0.000 | 0.363 | 0.370 |

**The A-vs-C test (does adding the rules help?), 5-fold GroupKFold:**

| Outcome | ΔR² (mean ± sd) | Sign stable? | Pre-registered bar (ΔR² ≥ 0.010) |
|---|--:|:-:|:-:|
| Reach | +0.0006 ± 0.0004 | no (one fold negative) | not cleared (~15× short) |
| Reception | +0.0000 ± 0.0004 | no (flips) | not cleared |

Single-split paired test on per-book errors: reach not significant (p ≈ 0.11), reception
negative (p ≈ 0.005).

**Per-claim (Model C, standardized, HC3):** single focal point −0.021\* on reach (≈ −5% per
SD, *wrong direction*, survives Bonferroni); thumbnail legibility −0.007\* / −0.006\* (*wrong
direction, both outcomes*); contrast +0.011\* / +0.003\* (right direction, trivial); low
clutter −0.009\* on reach (right direction, trivial); text-block count +0.006 (n.s.). `*` =
p < 0.05.

## Analysis

- **Joint outcome pattern:** neither outcome moves. Reach gets a lift too small to rely on and
  not reliably positive across folds; reception gets nothing.
- **What would have changed the conclusion:** a held-out ΔR² ≥ 0.010 on either outcome, or
  any single rule with an effect ≥ 12% per SD in the direction the advice predicts. Largest
  in-the-right-direction effect observed ≈ 2.5% per SD (contrast, reach); largest of any kind
  ≈ −5% (focal point), pointing the wrong way.
- **It holds under stricter tests too** — the solo-author / low-fame subset; the 1995-onward
  subset (book-age-control check); a blurb baseline (a TF-IDF read of the description carries
  *more* signal than the cover, and the cover still adds nothing on top); and within-author
  fixed effects (all effects still small — but the per-rule detail shifts: backwards
  focal-point halves, backwards thumbnail-on-reach fades, and **low clutter becomes the
  largest single effect, ~4%/SD, in the advice's direction**).
- **Limitations:** predictive, not causal (can't say "redesigning won't help you", only that
  rule-compliance doesn't separate winners from losers here); the **two-font rule was not
  properly operationalized** (text-block count ≠ typeface count) — read as "not tested", not
  "tested and failed"; rating count is **cumulative over the book's whole life**, so a
  launch-window cover effect is diluted to near-nothing and the data has no month-by-month
  history to recover it; Goodreads thumbnail-resolution images; English-language only; outcome
  is Goodreads ratings, downstream of price, blurb, and discovery algorithms; "best books ever"
  list, so everything already cleared a bar; thumbnail-legibility is the weakest measured
  feature (text *presence*, not *readability*).

See [`docs/modeling.md`](docs/modeling.md) for the full method and [`build-log.md`](build-log.md)
(November section) for every number and dead end.

---

## Running it yourself

Everything is reproducible from a clean checkout. The how-to lives in `docs/`:

- [`docs/setup.md`](docs/setup.md) — environment (there's a non-obvious `visual-clutter` install)
- [`docs/data.md`](docs/data.md) — getting and validating the dataset
- [`docs/pipeline.md`](docs/pipeline.md) — the 5 features + CLIP, running extraction (smoke and full)
- [`docs/outputs.md`](docs/outputs.md) — every file the analysis consumes, with schema
- [`docs/modeling.md`](docs/modeling.md) — the November benchmark: the four models, the split, the pre-registered bar, how to run it
- [`notebooks/`](notebooks/) — exploratory analysis (not on the reproducibility path)
