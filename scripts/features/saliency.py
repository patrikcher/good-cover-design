"""Claim: "single focal point / visual hierarchy".
Feature: concentration of a bottom-up saliency map.

Saliency model = DeepGaze IIE (Linardos et al.), trained on real human eye-tracking data --
predicts *where people actually look*, not just "where is there local contrast". Deployed
2026-09-05 replacing the original spectral-residual (Hou & Zhang 2007) implementation, which
was measured to (a) correlate +0.38 with an "empty background" proxy -- it read a mostly-blank
dark cover as having a strong focal point, and (b) near-duplicate the clutter metric (spearman
-0.88 with subband_entropy), undermining the "5 independent claims" the project tests.

One replacement was tried and rejected first: U2Net (salient-*object* segmentation, the kind
used for background removal). Its numbers looked like an improvement too, but it returns a
near-blank mask on any cover without a discrete photographic/illustrated object (text-only,
abstract, scenic covers) -- it isn't measuring something better on those, it isn't measuring
anything coherent. Only a visual check caught this; see scripts/analysis/saliency_model_ab.py
and build-log.md 2026-09-05 for the full evidence trail (both attempts, quantified + visual).

Cost of the swap: DeepGaze has float64 buffers that crash on MPS (Apple Silicon GPU) -- CPU
only, ~1.4 s/cover. Spectral residual is kept as a transparent secondary column (`*_sr` suffix)
precisely because it's nearly free to compute alongside -- not because it's still trusted as
the primary signal.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import requests
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.special import logsumexp

from scripts.cover_io import to_gray

_SR_SIZE = 128  # spectral residual work size (Hou & Zhang use ~64 on the short side)

_CENTERBIAS_URL = "https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy"
_CENTERBIAS_PATH = Path(os.path.expanduser("~/.cache/deepgaze/centerbias_mit1003.npy"))
_DEVICE = "cpu"  # see module docstring -- DeepGazeIIE's float64 buffers are not MPS-compatible

_model = None
_centerbias_template = None


# ---- primary: DeepGaze IIE -----------------------------------------------------------------
def _ensure_centerbias() -> np.ndarray:
    global _centerbias_template
    if _centerbias_template is None:
        if not _CENTERBIAS_PATH.exists():
            _CENTERBIAS_PATH.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(_CENTERBIAS_URL, timeout=60)
            r.raise_for_status()
            _CENTERBIAS_PATH.write_bytes(r.content)
        _centerbias_template = np.load(_CENTERBIAS_PATH)
    return _centerbias_template


def _get_model():
    global _model
    if _model is None:
        import deepgaze_pytorch

        _model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(_DEVICE).eval()
    return _model


def deepgaze_map(img: np.ndarray) -> np.ndarray:
    """DeepGaze IIE fixation-density prediction, returned at the input's H x W. Sums to ~1
    (it's a probability density), not normalized to [0, 1] -- callers that need a [0,1] display
    map should rescale."""
    import torch

    model = _get_model()
    cb_template = _ensure_centerbias()
    h, w = img.shape[:2]
    from scipy.ndimage import zoom

    cb = zoom(cb_template, (h / cb_template.shape[0], w / cb_template.shape[1]), order=0, mode="nearest")
    cb = cb - logsumexp(cb)
    img_t = torch.tensor(np.ascontiguousarray(img.transpose(2, 0, 1))[None]).float().to(_DEVICE)
    cb_t = torch.tensor(cb[None]).float().to(_DEVICE)
    with torch.no_grad():
        log_density = model(img_t, cb_t)
    return torch.exp(log_density[0, 0]).cpu().numpy()


# ---- secondary/legacy: spectral residual ----------------------------------------------------
def spectral_residual_map(img: np.ndarray) -> np.ndarray:
    """Spectral-residual saliency, returned at the input's H x W, normalized to [0, 1].
    Kept as the `_sr` secondary column -- see module docstring for why it was replaced."""
    g = to_gray(img)
    h, w = g.shape
    scale = _SR_SIZE / max(h, w)
    sh, sw = max(8, round(h * scale)), max(8, round(w * scale))
    ys = (np.linspace(0, h - 1, sh)).astype(int)
    xs = (np.linspace(0, w - 1, sw)).astype(int)
    small = g[np.ix_(ys, xs)]

    fft = np.fft.fft2(small)
    log_amp = np.log(np.abs(fft) + 1e-8)
    phase = np.angle(fft)
    spectral_residual = log_amp - uniform_filter(log_amp, size=3)
    recon = np.fft.ifft2(np.exp(spectral_residual + 1j * phase))
    sal = gaussian_filter(np.abs(recon) ** 2, sigma=3.0)

    sal = sal[np.ix_(
        np.clip((np.arange(h) * scale).astype(int), 0, sh - 1),
        np.clip((np.arange(w) * scale).astype(int), 0, sw - 1),
    )]
    sal -= sal.min()
    mx = sal.max()
    return sal / mx if mx > 0 else sal


# backwards-compatible alias -- some analysis scripts (saliency_model_ab.py, saliency_redundancy.py)
# import this name directly to mean "the classical method".
saliency_map = spectral_residual_map


def _gini(x: np.ndarray) -> float:
    x = np.sort(x.ravel())
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * x)) / (n * x.sum()))


def _concentration(flat: np.ndarray) -> tuple[float, float]:
    total = flat.sum()
    k = max(1, int(0.10 * flat.size))
    top10 = float(np.sort(flat)[-k:].sum() / total) if total > 0 else 0.0
    return _gini(flat), top10


def saliency_concentration(img: np.ndarray) -> dict[str, float]:
    """Higher gini / top-frac => a small region carries most of the predicted attention.

    sal_gini / sal_top10_mass: primary, from DeepGaze IIE.
    sal_gini_sr / sal_top10_mass_sr: secondary, from spectral residual -- kept for transparency
    and because it's nearly free to compute alongside; not used as the headline feature."""
    gini, top10 = _concentration(deepgaze_map(img).ravel())
    gini_sr, top10_sr = _concentration(spectral_residual_map(img).ravel())
    return {
        "sal_gini": gini,
        "sal_top10_mass": top10,
        "sal_gini_sr": gini_sr,
        "sal_top10_mass_sr": top10_sr,
    }
