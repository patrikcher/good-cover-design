"""Recompute just the saliency columns in features_full.parquet with the deployed DeepGaze
method (scripts/features/saliency.py) -- the other 11 columns (OCR, clutter) are untouched, so
this is its own targeted, resumable/shardable job, not a repeat of the full extraction run.

  python -m scripts.recompute_saliency
  # or sharded across 4 terminals:
  python -m scripts.recompute_saliency --shard 0/4
  python -m scripts.recompute_saliency --shard 1/4   # ...2/4, 3/4
  python -m scripts.recompute_saliency --merge        # combine the 4 shards + write features_full.parquet

~1.4-1.6 s/cover (CPU only, DeepGaze has no MPS path) -> ~16-18h single-thread for the full
42,344, ~4-6h sharded 4-way. See build-log.md 2026-09-05 for why this replaced spectral residual.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
import zlib
from pathlib import Path

import pandas as pd

from scripts.cover_io import load_cover
from scripts.features.saliency import saliency_concentration

FEATURES = "data/features_full.parquet"
COVERS = Path("data/covers")


def _shard_path(shard: str | None) -> Path:
    base = Path("data/saliency_recompute.jsonl")
    if not shard:
        return base
    i, n = shard.split("/")
    return base.with_suffix(f".shard{i}of{n}.jsonl")


def _merge() -> None:
    p = Path("data/saliency_recompute.jsonl")
    shards = sorted(p.parent.glob(f"{p.stem}.shard*of*{p.suffix}"))
    sources = shards if shards else ([p] if p.exists() else [])
    if not sources:
        raise SystemExit(f"nothing to merge: no {p} and no {p.stem}.shard*of*{p.suffix}")

    rows: dict[str, dict] = {}
    for s in sources:
        for line in s.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["key"]] = r
    label = f"{len(shards)} shards" if shards else p.name
    print(f"merged {label} -> {len(rows)} rows")

    feat = pd.read_parquet(FEATURES)
    cols = ["sal_gini", "sal_top10_mass", "sal_gini_sr", "sal_top10_mass_sr"]
    updated = 0
    for key, r in rows.items():
        if key in feat.index:
            for c in cols:
                feat.loc[key, c] = r[c]
            updated += 1
    feat.to_parquet(FEATURES)
    print(f"updated {updated} rows in {FEATURES} (of {len(feat)} total)")
    missing = set(feat.index) - set(rows)
    if missing:
        print(f"{len(missing)} rows not recomputed (no matching key) -- old spectral-residual "
              f"values retained for these; rerun the missing shard(s) if that's unexpected")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", help="i/N -- process only crc32(key)%%N==i")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--checkpoint", type=int, default=100)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.merge:
        _merge()
        return

    feat = pd.read_parquet(FEATURES)
    keys = feat.index.astype(str).tolist()
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        keys = [k for k in keys if zlib.crc32(k.encode()) % n == i]
    if args.limit:
        keys = keys[: args.limit]

    out = _shard_path(args.shard)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():
        with out.open() as f:
            done = {json.loads(line)["key"] for line in f if line.strip()}
        print(f"resume: {len(done)} keys already done in {out.name}")

    todo = [k for k in keys if k not in done]
    print(f"{len(todo)} covers to process")

    fails, t0 = [], time.time()
    with out.open("a") as sink:
        for i, key in enumerate(todo, 1):
            try:
                img = load_cover(str(COVERS / f"{key}.jpg"))
                r = saliency_concentration(img)
                r["key"] = key
                sink.write(json.dumps(r) + "\n")
            except Exception as e:  # noqa: BLE001
                fails.append((key, repr(e)))
                if len(fails) <= 5:
                    traceback.print_exc()
            if i % args.checkpoint == 0 or i == len(todo):
                sink.flush()
                print(f"  {i}/{len(todo)}  {(time.time()-t0)/i:.2f}s/img  {len(fails)} fail")

    if fails:
        fp = out.with_suffix(".failures.csv")
        pd.DataFrame(fails, columns=["key", "error"]).to_csv(fp, index=False)
        print(f"{len(fails)} failures -> {fp}")


if __name__ == "__main__":
    main()
