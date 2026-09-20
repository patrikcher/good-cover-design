# notebooks/ — exploration only

These notebooks are **not on the reproducibility path.** They are scratchpads for looking at
distributions and sanity-checking features by eye.

Rule for this repo: every number that ends up in an article or that a design decision rests on
must also exist as a runnable script under `scripts/` or `scripts/analysis/`, and be logged in
`build-log.md`. A notebook is never the sole source of a load-bearing number.

Notebooks:
- `week2_feature_eda.ipynb` — feature distributions + collinearity on the 50-cover sample
- `week3_dataset_eda.ipynb` — English-filter effect, author-fame vs outcomes, CLIP structure by
  genre (sections 1–3 run on current data; section 4 needs `features_full.parquet`)
- `architecture_choices_eda.ipynb` — runs the four `scripts/analysis/*` checks behind the
  Teardown article's architecture-choice discussion together in one place, with the figures
  inline. Each section is a thin wrapper around its script (see below); the notebook adds
  nothing load-bearing of its own.

Current load-bearing checks that were promoted out of notebooks into scripts:
- `scripts/analysis/saliency_redundancy.py` — the spearman −1.0 that justified dropping `sal_norm_entropy`
- `scripts/analysis/saliency_model_ab.py` — is the saliency method defensible? current vs. U2Net (rejected) vs. DeepGaze (validated, not deployed) — dev-only deps, see its docstring
- `scripts/analysis/thumbnail_legibility_ab.py` — v1 (char retention) vs v2 (box retention), the redesign evidence
- `scripts/analysis/author_fame_leakage.py` — what the naive (non-leave-one-out) fame proxy would have leaked
- `scripts/analysis/ocr_engine_ab.py` — EasyOCR vs. Tesseract, tested rather than asserted
- `scripts/smoke_features_report.py` — sample feature distributions + collinearity scan + contact sheet

## Running

```
.venv/bin/python -m pip install jupyter   # not in requirements-frozen.txt; dev-only
.venv/bin/jupyter lab notebooks/
```

Regenerate inputs first if `data/` is clean:
```
.venv/bin/python -m scripts.fetch_sample
.venv/bin/python -m scripts.extract_features --sample-dir data/sample_covers --out data/features_sample.parquet
```
