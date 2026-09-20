"""CLIP image embedding -- the catch-all diagnostic feature.

Not an interpretable "here's the rule" result. In the November modeling it's Model D: does a
generic learned image representation predict success beyond what the 5 named features capture?
An upper bound on "signal in the cover", nothing more.

Model: open_clip ViT-B-32 / laion2b_s34b_b79k (512-dim). B-32 over L-14 for throughput on
the full 42k run; swap via env CLIP_MODEL / CLIP_PRETRAINED if you want the bigger one.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

_MODEL = os.environ.get("CLIP_MODEL", "ViT-B-32")
_PRETRAINED = os.environ.get("CLIP_PRETRAINED", "laion2b_s34b_b79k")
_DIM = {"ViT-B-32": 512, "ViT-L-14": 768}.get(_MODEL, 512)

_model = None
_preprocess = None
_device = "mps" if torch.backends.mps.is_available() else "cpu"


def _load():
    global _model, _preprocess
    if _model is None:
        import open_clip

        _model, _, _preprocess = open_clip.create_model_and_transforms(
            _MODEL, pretrained=_PRETRAINED
        )
        _model = _model.to(_device).eval()
    return _model, _preprocess


def embed_batch(imgs: list[np.ndarray]) -> np.ndarray:
    """imgs: list of HxWx3 uint8 (any size -- CLIP's own preprocess handles resize/crop).
    Returns (len(imgs), DIM) float32, L2-normalized."""
    model, preprocess = _load()
    batch = torch.stack([preprocess(Image.fromarray(im)) for im in imgs]).to(_device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)


def embed_one(img: np.ndarray) -> np.ndarray:
    return embed_batch([img])[0]


EMBED_DIM = _DIM
