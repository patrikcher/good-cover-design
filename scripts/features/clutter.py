"""Claim: "low clutter".
Feature: Rosenholtz Feature Congestion + Subband Entropy, via the `visual-clutter` package.

Week 1 smoke test cleared this as the headline clutter feature (50/50, no fallback needed) --
see build-log.md 2026-09-04 and requirements-frozen.txt for the unsupported-but-working dep set.
Vlc reads from a file path and writes intermediates to `output_dir`, so this wrapper stages a
temp PNG and points output at scratch.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

_SCRATCH = Path(tempfile.gettempdir()) / "vc_out"


def clutter_metrics(img: np.ndarray) -> dict[str, float]:
    from visual_clutter import Vlc

    _SCRATCH.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tf:
        Image.fromarray(img).save(tf.name)
        clt = Vlc(tf.name, numlevels=3, contrast_filt_sigma=1, color_pool_sigma=3,
                  output_dir=str(_SCRATCH), prefix="vc")
        fc, _ = clt.getClutter_FC()
        se = clt.getClutter_SE()
    return {"feature_congestion": float(fc), "subband_entropy": float(se)}
