# Setup

Built and run on macOS (Apple Silicon), Python 3.12. Linux should work; Windows won't
(`visual-clutter` / `pyrtools` don't support it).

## 1. Virtualenv + dependencies

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install --no-deps visual-clutter==1.0.7
.venv/bin/pip install -r requirements-frozen.txt
```

`requirements-frozen.txt` is the exact working set, including two git-installed packages
(`clip`, `deepgaze_pytorch` — pinned to a commit, since neither has a PyPI release); a plain
`pip install -r` handles git URLs fine, no separate step needed. Everything runs as a module
from the repo root:

```bash
.venv/bin/python -m scripts.<name>        # NOT python scripts/<name>.py
```

(`scripts/__init__.py` has to load first — see §3.)

Dev-only extras for the notebooks: `.venv/bin/pip install -r requirements-dev.txt`.

## 2. The `visual-clutter` install is deliberately unsupported

`visual-clutter` (Rosenholtz Feature Congestion + Subband Entropy, our "low clutter" feature)
was last released Aug 2023 and hard-pins `numpy<1.25, scipy<1.10, Pillow<9.0,
opencv-python<4.6, scikit-image<0.21`. None of those old versions build on Python 3.12.

The fix is `--no-deps` + current versions of everything (see step 1). Smoke-tested clean on
50 covers, twice — see `build-log.md`. **Risks you're accepting:**

- The dep versions are far outside the package's declared range. If a future `pip install`
  resolves them differently, it can break silently. Pin exactly (that's what
  `requirements-frozen.txt` is for); don't float.
- `pyrtools` (the C dependency people worried about) ships an arm64 wheel — it was a
  non-issue in practice.
- The output isn't validated against the reference MATLAB implementation numerically. We use
  Feature Congestion / Subband Entropy as *relative* features across covers, not absolute
  measurements, so this is acceptable.

If it ever does break, the planned fallback order: reimplement Subband Entropy with
`pywavelets`; then grayscale-histogram entropy + JPEG compression ratio as a cruder proxy,
labelled as such.

## 3. TLS: `SSL_CERT_FILE`

The system Python on the build machine has no usable CA bundle, so anything reaching the
network through stdlib `urllib` (EasyOCR's model download, some torch paths, raw cover fetches
via `urllib`) dies with `CERTIFICATE_VERIFY_FAILED`.

`scripts/__init__.py` sets `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` to `certifi.where()` before
any networked import. **Any script that touches the network must `import scripts` first** —
the existing ones already do. If you call the feature code from your own entry point, import
the package before importing anything that hits the net.

## 4. OpenCV: two packages coexist

EasyOCR pulls `opencv-python-headless`; we also have `opencv-python`. Both resolve at the
pinned version with no observed conflict. Noted as a latent risk, not an active problem.

## 5. First run downloads models

- `scripts/extract_features.py` → EasyOCR downloads its CRAFT detector + CRNN recognizer
  (~100 MB) to `~/.EasyOCR/` on first use.
- `scripts/extract_clip.py` → `open_clip` downloads ViT-B-32 / laion2b weights (~600 MB) from
  the HuggingFace Hub to `~/.cache/huggingface/`. The "unauthenticated requests to the HF Hub"
  warning is harmless; set `HF_TOKEN` only if you hit rate limits.
- `scripts/features/saliency.py` (DeepGaze, the saliency feature) → downloads its own backbone
  weights (~900 MB across `densenet201`, `efficientnet-b5`, `resnext50`, a finetuned resnet50,
  and the DeepGaze ensemble itself) to `~/.cache/torch/hub/checkpoints/` on first use, plus an
  8 MB centerbias prior to `~/.cache/deepgaze/`.

All one-time. All need the `SSL_CERT_FILE` fix from §3 (already handled by `scripts/__init__.py`).

## 6. DeepGaze is CPU-only — no MPS path

The saliency feature's model (DeepGaze IIE) keeps some buffers in float64, which Apple's MPS
backend cannot hold (`Cannot convert a MPS Tensor to float64 dtype`) — a hard crash, not a slow
fallback. `scripts/features/saliency.py` hardcodes `_DEVICE = "cpu"`. Cost: ~1.2–1.6 s/cover
single-process, so a full-corpus run (`scripts/recompute_saliency.py`) is ~13 h. **Do not
shard it** — 4 DeepGaze copies oversubscribe the fast cores and slow to ~14 s/cover, worse than
one process. DeepGaze replaced spectral residual (near-instant,
pure math, still kept as the `_sr` secondary column) because spectral residual was measured to
confuse empty backgrounds for a focal point and near-duplicate the clutter metric; see
build-log.md 2026-09-05 and `scripts/analysis/saliency_model_ab.py` for the evidence trail.
