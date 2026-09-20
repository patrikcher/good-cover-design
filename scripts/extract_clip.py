"""Compute CLIP image embeddings for the working set -> data/clip_embeddings.npz.

  python -m scripts.extract_clip                    # all covers on disk
  python -m scripts.extract_clip --limit 500 --out data/clip_sample.npz

npz contents: keys (U32 array), embeddings (float32, N x EMBED_DIM, L2-normalized).
Resumable: existing keys in --out are kept and skipped.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import scripts  # noqa: F401
from scripts.cover_io import load_cover
from scripts.features.clip_embed import EMBED_DIM, embed_batch

BOOKS = "data/books_english.parquet"
COVERS = Path("data/covers")


def _load_existing(out: Path) -> dict[str, np.ndarray]:
    if not out.exists():
        return {}
    z = np.load(out, allow_pickle=True)
    return dict(zip(z["keys"].tolist(), z["embeddings"]))


def _save(out: Path, store: dict[str, np.ndarray]) -> None:
    # keys vary in length (bookId slugs run to ~130 chars) -> object dtype, not fixed-width
    keys = np.array(list(store), dtype=object)
    emb = np.stack(list(store.values())).astype(np.float32) if store else np.zeros((0, EMBED_DIM), np.float32)
    np.savez(out, keys=keys, embeddings=emb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--checkpoint", type=int, default=1024)
    ap.add_argument("--out", default="data/clip_embeddings.npz")
    args = ap.parse_args()
    out = Path(args.out)

    keys = pd.read_parquet(BOOKS).index.astype(str).tolist()
    if args.limit:
        keys = keys[: args.limit]

    store = _load_existing(out)
    print(f"resume: {len(store)} embeddings already in {out.name}")
    todo = [k for k in keys if k not in store and (COVERS / f"{k}.jpg").exists()]
    print(f"{len(todo)} covers to embed  (dim {EMBED_DIM})")

    buf_keys: list[str] = []
    buf_imgs: list[np.ndarray] = []
    n_since_ckpt = 0

    def flush():
        nonlocal n_since_ckpt
        if not buf_keys:
            return
        for k, v in zip(buf_keys, embed_batch(buf_imgs)):
            store[k] = v
        n_since_ckpt += len(buf_keys)
        buf_keys.clear()
        buf_imgs.clear()

    for i, k in enumerate(todo, 1):
        try:
            buf_imgs.append(load_cover(str(COVERS / f"{k}.jpg")))
            buf_keys.append(k)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {k}: {e!r}")
        if len(buf_keys) >= args.batch:
            flush()
        if n_since_ckpt >= args.checkpoint:
            _save(out, store)
            n_since_ckpt = 0
            print(f"  {i}/{len(todo)}  ({len(store)} stored)")

    flush()
    _save(out, store)
    print(f"wrote {out}  ({len(store)} x {EMBED_DIM})")


if __name__ == "__main__":
    main()
