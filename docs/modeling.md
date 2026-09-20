# Modeling — the November benchmark

The question: do the five confidently-stated cover-design rules (single focal point, high
text/background contrast, two-font maximum, thumbnail legibility, low clutter) predict a
book's success once you control for **author fame** and **genre** — or is Peter Mendelsund's
"there is no science to this" closer to right?

Everything here runs from the repo root as `python -m scripts.modeling.<name>` and consumes
the pipeline outputs described in [`outputs.md`](outputs.md).

## Run order

```bash
# one-off: the ViT-L-14 embedding Model D uses as its ceiling (Part 1 only builds ViT-B-32)
CLIP_MODEL=ViT-L-14 CLIP_PRETRAINED=laion2b_s32b_b82k \
  python -m scripts.extract_clip --out data/clip_embeddings_l14.npz

python -m scripts.modeling.assemble          # -> data/model_frame.parquet
python -m scripts.modeling.split             # -> data/model_split.parquet
python -m scripts.modeling.models            # -> data/model_results.json, data/model_c_coefficients.csv
python -m scripts.modeling.robustness        # Week 2 checks -> stdout + build-log
python -m scripts.modeling.generalization    # Week 3 checks -> data/generalization_results.json
python -m scripts.modeling.blurb_check       # does the cover beat the blurb? -> stdout
python -m scripts.modeling.within_author     # within-author fixed effects -> data/within_author_coefficients.csv
python -m scripts.modeling.comparison_table  # -> data/model_comparison.csv, article-assets/model-comparison.md
```

`assemble`, `split`, and the L-14 embedding are deterministic and only need re-running if the
working set changes. `models.py` auto-uses `data/clip_embeddings_l14.npz` when present and
falls back to the ViT-B-32 `data/clip_embeddings.npz` otherwise.

## The frame (`assemble.py`)

One row per cover, 42,344 rows (`books_english` ⋈ `features_full` inner — one book has no
feature row — ⋈ `author_fame` exact).

| Group | Columns |
|---|---|
| Outcomes (kept separate, never blended) | `y_log_ratings` = log10(numRatings); `y_rating` = average rating. Raw `numRatings` / `rating` retained for the count-GLM check. |
| Author-fame controls (all leave-one-out) | `fame_log_sum_ratings`, `fame_log_max_ratings`, `fame_log_n_books`, `fame_mean_rating`, `author_is_solo` |
| Genre controls | `g_*` multi-hot for every Goodreads shelf tag on ≥ 1% of rows (**141 dummies**), `has_genre`. Single frequency threshold, no hand-curation — the set includes format/meta tags (`g_audiobook`, `g_novels`). Deliberate: a rich genre control makes Model C's incremental test hard. |
| Metadata controls (joined from the raw CSV) | `book_age` (from `firstPublishDate`, else `publishDate`; + `pub_year_missing`, `pub_year_suspect`); `has_series` + `log_series_size`; `self_published` + `publisher_missing` + **44 publisher dummies** (≥ 0.3% of rows). Book age is the largest of these — older books accumulate more ratings, and cover style is era-dependent, so it is the clearest non-cover confound for reach. |
| The 5 claim features | `sal_gini`; `title_contrast_ratio` (+ `_missing`, 2.5%); `text_block_count`; `legib_box_retention` (+ `_missing`, 0.3%); `feature_congestion` + `subband_entropy`. NaNs are flagged then median-imputed, never silently. |

**Book-age caveat:** the publish-date field mixes `MM/DD/YY`, `Month Nth YYYY`, and bare
`YYYY`; the slash form pivots at 26, so books published before 1927 read as recent. ~489
Classics are flagged `pub_year_suspect`; `generalization.py`'s 1995-onward subset check is the
real backstop.

Collinear twins (`sal_top10_mass`, `median_text_contrast_ratio`, `legib_char_retention`, the
`*_sr` saliency columns) are left out of the frame and kept in `features_full.parquet` for
robustness checks.

## The split (`split.py`)

`GroupShuffleSplit` on `primary_author`, seed 42, 80/20 → **train 33,845 / test 8,499, zero
author overlap**. 71% of books share an author with another book in the set, so a random
split would let a model — Model D especially — be rewarded for learning an author's visual
style. The leave-one-out fame covariates already block the fame-*number* leak; grouping
blocks the style leak. Written once to `data/model_split.parquet`; every model reads it.

## The four models (`models.py`)

OLS with HC3 robust errors, on both outcomes separately. Continuous predictors are
standardized on **train** statistics. The two Rosenholtz clutter columns (train corr 0.72)
are averaged into a single `clutter_z` — entered separately they split into an
equal-and-opposite suppression pair, and "low clutter" is one claim.

| Model | Predictors | Role |
|---|---|---|
| **A** baseline | genre + author fame + book age + series + publisher | how much of success is explained without the cover at all |
| **B** cover-only | the 5 claim features, no controls | the naive/confounded picture — diagnostic, not a headline |
| **C** controlled | A + the 5 claim features | **the real test**: does the cover add anything beyond the baseline? |
| **D** CLIP | A + 100 PCA components of the **ViT-L-14** CLIP embedding (`data/clip_embeddings_l14.npz`; `models.py` auto-picks it, override with `CLIP_NPZ`) | upper bound on *any* cover signal; compared to C, not A. Part 1's pipeline ships a ViT-B-32 embedding as the standard catch-all; the ceiling uses the larger model on purpose |

**The A-vs-C test:** a robust (HC3) Wald F-test on the joint nullity of the feature block
(train), plus the held-out check the verdict actually rests on — ΔR² / ΔRMSE on the test
set, an author-clustered paired test on squared errors, and a bootstrap CI over test
authors. `generalization.py` adds a 5-fold `GroupKFold` version so the ΔR² isn't one
lucky/unlucky split.

## Pre-registered bar

Fixed before reading Model C. The rules "matter" if **either**:

- (a) adding the 5 features lifts held-out R² by ≥ **0.010** on either outcome, or
- (b) some single claim shows a held-out-robust standardized effect ≥ **0.05** in the
  direction the advice predicts.

Neither is a high bar — (a) is roughly the CLIP upper bound's entire margin over baseline.

## Robustness (`robustness.py`) and generalization (`generalization.py`)

1. **Combined clutter z-index** — confirms the two-column suppression and that the block
   conclusion is unchanged.
2. **Count GLM** — Gamma(log) on raw `numRatings` instead of OLS-on-log10; A-vs-C lands the
   same.
3. **Within-genre** — Model C per genre for the six biggest; the pooled null isn't averaging
   away a within-genre effect. (Controls trimmed per subset — the full genre-dummy set is
   rank-deficient inside one genre.)
4. **Low-fame subset** — solo-author and low-fame books; the null holds where the advice
   should bite hardest. Plus a feature × `author_is_solo` interaction test.
5. **Grouped 5-fold** — is the held-out ΔR² sign stable across splits?
6. **Nonlinearity** — quadratic terms for the two features with real coefficients, plus a
   decile table; the advice is about thresholds, so linear-only would be a fair objection.
7. **Modern subset (1995+)** — reruns the A-vs-C + per-claim test on books with unambiguous
   publish dates, backstopping the book-age control.
8. **Blurb baseline** (`blurb_check.py`) — adds a TF-IDF/SVD compression of the book's
   `description` to the baseline. The blurb carries more predictive signal than the cover
   (+0.005 reach / +0.013 rating vs the cover's +0.001 / −0.001); the cover still adds nothing
   on top of it.
9. **Within-author fixed effects** (`within_author.py`) — author dummies via within-author
   mean-subtraction, so every stable author trait is absorbed. Multi-book authors only
   (30k books / 6k authors), in-sample, SEs clustered by author, no held-out split. Big
   picture holds (all effects small); the per-rule detail shifts — focal point stays backwards
   at ~half size, thumbnail-on-reach loses significance, and **low clutter becomes the largest
   effect (~4%/SD) and points the advice's way**.

## Result

See [`build-log.md`](../build-log.md) (November section) for every number and
`article-assets/model-comparison.md` for the rendered tables. In short: controlling for genre,
author fame, book age, series, and publisher, the five rules add **no reach improvement that
survives re-splitting** (5-fold ΔR² ≈ +0.0006, one fold negative, single-split paired test
n.s.) and **nothing to average rating**. Individually: focal-point concentration **−5%/SD**
and thumbnail legibility both significant and pointing **against** the advice (and holding on
the 1995+ subset and under multiple-testing correction); contrast **+2.5%/SD** and clutter
**−2%/SD** correct-signed but trivial; text-block count n.s. The CLIP upper bound (ViT-L-14)
adds ≈ 0.7% over the same baseline. Mendelsund's side, quantified.
