# Feature pipeline

Extracts one row of interpretable visual features per cover, each mapped to a specific,
testable cover-design claim, plus a CLIP embedding as a catch-all diagnostic.

```
                        ┌─ saliency.py   DeepGaze IIE (+ spectral-residual, secondary `_sr` cols)
cover_io.load_cover ────┼─ text.py       one EasyOCR pass at 480px + one at 160px
(strip suffix, RGB,     └─ clutter.py    visual-clutter (Vlc) on a temp PNG
 resize to 480px tall)  ────────────────────────────────────> features_full.parquet  (extract_features.py)

cover_io.load_cover ───── clip_embed.py  open_clip ViT-B-32 ─> clip_embeddings.npz    (extract_clip.py)
```

`scripts/cover_io.py` is the shared front door — every extractor consumes its normalized
480 px output so resolution is never a confound. The interpretable features and CLIP run as
two separate scripts (CPU per-image vs GPU-batched), both keyed by `bookId`, both resumable.

## The 5 claims → features

| Claim (from self-publishing / author-services blogs) | Feature column(s) | Method |
|---|---|---|
| Single focal point / visual hierarchy | `sal_gini`, `sal_top10_mass` | DeepGaze IIE fixation-prediction density (trained on real eye-tracking data); Gini + top-10%-mass concentration. Spectral residual (Hou & Zhang 2007) kept as secondary `sal_gini_sr`/`sal_top10_mass_sr` — see caveats below |
| High text/background contrast | `title_contrast_ratio`, `median_text_contrast_ratio` | OCR the title box, Otsu-split ink vs paper inside it, WCAG contrast ratio |
| Two-font maximum / typographic simplicity | `text_block_count` | Union-find merge of OCR lines into blocks; count of blocks (a region count, not a font count — rough proxy, as intended) |
| Thumbnail legibility | `legib_box_retention` (primary), `legib_char_retention` (secondary) | Ratio of CRAFT text-box count at 160 px vs 480 px |
| Low clutter | `feature_congestion`, `subband_entropy` | Rosenholtz metrics via `visual-clutter` |

Peter Mendelsund and AIGA are **not** a 6th source — they argue no formula should exist, which
is the position the whole project tests against, not a feature.

**Plus the catch-all:** `clip_embed.py` → a 512-dim open_clip ViT-B-32 embedding per cover
(`data/clip_embeddings.npz`). Not an interpretable rule — it's the Model D upper bound on
"is there *any* signal in the cover beyond the 5 named features". Swap to ViT-L-14 via
`CLIP_MODEL` / `CLIP_PRETRAINED` env vars if you want the bigger model.

## Output schema (`data/features_*.parquet`)

Index: `key` (the `bookId`, or the sample filename stem).

| Column | Type | Meaning |
|---|---|---|
| `img_h`, `img_w` | int | normalized dims; `img_h` is always 480 |
| `sal_gini` | float | Gini of the DeepGaze fixation-density map. High → predicted attention concentrated in few pixels. |
| `sal_top10_mass` | float | fraction of total predicted attention in the brightest 10% of pixels |
| `sal_gini_sr`, `sal_top10_mass_sr` | float | same two stats from spectral residual, the original (replaced) method — secondary, transparency column |
| `text_block_count` | int | merged OCR text blocks (0 if no text) |
| `n_text_detections` | int | raw OCR detections above conf 0.30 |
| `ocr_char_count` | int | alphanumeric chars recognized at full (480 px) res |
| `n_text_boxes_full` | int | CRAFT detector boxes at 480 px |
| `legib_box_retention` | float | `n_text_boxes(160px) / n_text_boxes(480px)`, clipped [0,1] |
| `legib_char_retention` | float | `chars_ocr(160px) / chars_ocr(480px)` — **known noisy**, kept for transparency |
| `legib_conf_ratio` | float | mean OCR confidence at 160 px ÷ at 480 px (NaN if either pass finds nothing) |
| `title_contrast_ratio` | float | WCAG ratio (1–21) for the largest OCR text box; **NaN if no text on the cover** |
| `median_text_contrast_ratio` | float | median WCAG ratio across all text boxes |
| `feature_congestion` | float | Rosenholtz Feature Congestion |
| `subband_entropy` | float | Rosenholtz Subband Entropy |

## Running extraction

```bash
# smoke test — 50 sample covers, ~2.5s/cover
python -m scripts.fetch_sample
python -m scripts.extract_features --sample-dir data/sample_covers --out data/features_sample.parquet
python -m scripts.smoke_features_report      # contact sheet + distributions + collinearity scan
```

### Full run (~42k English covers)

```bash
python -m scripts.build_dataset              # -> data/books_english.parquet  (once)
python -m scripts.author_fame                # -> data/author_fame.parquet    (once, seconds)
python -m scripts.download_covers            # -> data/covers/<key>.jpg       (~13 min, 8 workers)

# interpretable features — the slow part. ~4 s/cover (EasyOCR + DeepGaze) => ~47 h single-thread.
python -m scripts.extract_features --books --out data/features_full.parquet
#   ...or shard across N terminals and merge:
python -m scripts.extract_features --books --out data/features_full.parquet --shard 0/4   # x4, i=0..3
python -m scripts.extract_features --books --out data/features_full.parquet --merge data/features_full.parquet

# CLIP embeddings — fast. ~35 min total.
python -m scripts.extract_clip               # -> data/clip_embeddings.npz
```

All three extraction commands are **resumable** — re-run after an interruption and they skip
finished keys. `extract_features` appends to `<out>.jsonl` and rewrites the parquet every
`--checkpoint` (100) rows. EasyOCR + DeepGaze on CPU are the bottleneck; the extractors are
independent per cover, so sharding scales linearly.

**If `features_full.parquet` already exists and only the saliency method changed** (this
happened once — spectral residual → DeepGaze, 2026-09-05), don't redo the whole run.
`scripts/recompute_saliency.py` updates only the 4 saliency columns in place, resumable,
~13h **single-process** for the full 42,344:
```bash
python -m scripts.recompute_saliency               # ~1.2s/cover, ~13h
python -m scripts.recompute_saliency --merge        # write into features_full.parquet
```
**Don't shard this one.** `--shard i/N` exists but 4-way DeepGaze copies oversubscribe the
machine's fast cores and drop to ~14s/cover — slower overall than one clean process. (The
October `extract_features` run sharded fine because EasyOCR is lighter per-process.)

See [`outputs.md`](outputs.md) for every file the modeling consumes.

## Per-feature caveats (carry into the modeling)

- **`sal_gini` was redesigned — deployed 2026-09-05.** The original method (spectral residual)
  fired on any high-contrast region, so a minimal dark cover scored as "one focal point" partly
  because it's mostly empty (+0.38 correlation with an empty-background proxy) and
  near-duplicated the clutter metric (spearman −0.88 with `subband_entropy`), undermining the
  "5 independent claims" framing. DeepGaze IIE (trained on real eye-tracking data) fixed both
  — confound correlation +0.38→−0.04, clutter correlation −0.88→−0.13 on the full 42,344
  (reduced, not suspiciously erased). Cost: CPU-only, no MPS path, ~1.2–1.6s/cover — ~13h for
  the full corpus single-process (sharding 4-way *hurt* here, unlike the OCR run: DeepGaze is
  heavier per-process and 4 copies oversubscribed the machine's fast cores, dropping to
  ~14s/cover). Spectral residual kept as
  `sal_gini_sr`/`sal_top10_mass_sr`. A first replacement attempt, U2Net (salient-object
  segmentation), was tried and **rejected** — its numbers looked better too, but it returns a
  near-blank mask on any cover without a discrete object; only a visual check caught that it
  wasn't measuring anything coherent on those. Full evidence: build-log.md 2026-09-05,
  `python -m scripts.analysis.saliency_model_ab`.
- **EasyOCR over Tesseract — tested, and the honest finding is narrower than first claimed.**
  Default-settings Tesseract misses text entirely on ~32% of covers EasyOCR reads; tuned to a
  fairer sparse-text PSM its raw character recall is competitive. What doesn't hold up under
  tuning is quality: on a decorative/cursive title it fragments the word into unreadable noise
  rather than text, which would inflate `text_block_count` and corrupt which region
  `title_contrast_ratio` picks as "the title." Repro: `python -m scripts.analysis.ocr_engine_ab`.
- **Thumbnail legibility was redesigned.** v1 (`legib_char_retention`) measured EasyOCR's
  recognizer floor at ~30 px cap-height, not the cover — 12/46 exact zeros, ~0 correlation
  with contrast. v2 (`legib_box_retention`) uses the detector, which degrades gracefully.
  v1 is retained as a transparently-noisy column. Repro:
  `python -m scripts.analysis.thumbnail_legibility_ab`.
- **Contrast is NaN for wordless covers.** 2.5% of covers at full N (1,049/42,342). Give the
  models an explicit missingness indicator; don't impute silently.
- **Collinearity, full 42,344, post-DeepGaze** (`python -m scripts.smoke_features_report
  --features data/features_full.parquet --no-sheet`): the primary `sal_gini` (DeepGaze) is no
  longer in the |spearman| > 0.6 list at all — `sal_gini`↔`feature_congestion` −0.23,
  `sal_gini`↔`subband_entropy` −0.13 (was −0.74 / −0.88 with spectral residual). The
  near-duplication of the clutter metric is gone; saliency concentration and clutter now
  measure different things. Pairs Model C still needs to handle: `sal_gini`↔`sal_top10_mass`
  0.96 (keep one), `title`↔`median` contrast 0.78, `n_text_detections`↔`ocr_char_count` 0.79,
  `feature_congestion`↔`subband_entropy` 0.73. The secondary `sal_gini_sr` still carries the
  old 0.88 collinearity with `subband_entropy` — that's expected, it's the deprecated method
  kept for transparency; don't feed both `sal_gini` and `sal_gini_sr` to the same model.
- **`sal_norm_entropy` was dropped** (spearman −0.997 with `sal_gini` — a restatement). Repro:
  `python -m scripts.analysis.saliency_redundancy`.
