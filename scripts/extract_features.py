"""Run the five interpretable feature extractors over a set of covers.

Smoke test (Week 2):
  python -m scripts.extract_features --sample-dir data/sample_covers --out data/features_sample.parquet

Full run (Week 3) -- reads data/books_english.parquet, covers from data/covers/<key>.jpg:
  python -m scripts.download_covers                       # fetch covers first
  python -m scripts.extract_features --books --out data/features_full.parquet
  # or shard across processes:
  python -m scripts.extract_features --books --out data/features_full.parquet --shard 0/4
  python -m scripts.extract_features --books --out data/features_full.parquet --shard 1/4  ...
  python -m scripts.extract_features --merge data/features_full.parquet   # combine shards

Resumable: appends to <out>.jsonl and rewrites <out> every --checkpoint rows; a rerun skips
keys already present. ~2.5 s/cover single-thread.
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
from scripts.features.clutter import clutter_metrics
from scripts.features.saliency import saliency_concentration
from scripts.features.text import text_features

BOOKS = "data/books_english.parquet"
COVERS = Path("data/covers")


def extract_one(src: str, cache_key: str | None = None) -> dict:
    img = load_cover(src, cache_key=cache_key)
    feats: dict[str, object] = {"img_h": img.shape[0], "img_w": img.shape[1]}
    feats.update(saliency_concentration(img))
    feats.update(text_features(img))
    feats.update(clutter_metrics(img))
    return feats


def _shard_path(out: str, shard: str | None) -> Path:
    if not shard:
        return Path(out)
    i, n = shard.split("/")
    p = Path(out)
    return p.with_suffix(f".shard{i}of{n}{p.suffix}")


def _targets(args) -> list[tuple[str, str]]:
    if args.sample_dir:
        return [(str(p), p.stem) for p in sorted(Path(args.sample_dir).glob("*.jpg"))]
    df = pd.read_parquet(BOOKS)
    keys = df.index.astype(str).tolist()
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        keys = [k for k in keys if zlib.crc32(k.encode()) % n == i]
    if args.limit:
        keys = keys[: args.limit]
    return [(str(COVERS / f"{k}.jpg"), k) for k in keys]


def _merge(out: str) -> None:
    p = Path(out)
    shards = sorted(p.parent.glob(f"{p.stem}.shard*of*{p.suffix}"))
    if not shards:
        raise SystemExit(f"no shards matching {p.stem}.shard*of*{p.suffix}")
    df = pd.concat([pd.read_parquet(s) for s in shards])
    df = df[~df.index.duplicated(keep="first")]
    df.to_parquet(p)
    print(f"merged {len(shards)} shards -> {p}  {df.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir")
    ap.add_argument("--books", action="store_true", help=f"read keys from {BOOKS}")
    ap.add_argument("--shard", help="i/N -- process only crc32(key)%%N==i")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--checkpoint", type=int, default=100)
    ap.add_argument("--out")
    ap.add_argument("--merge", help="combine <out>.shard*.parquet into this path and exit")
    args = ap.parse_args()

    if args.merge:
        _merge(args.merge)
        return
    if not args.out:
        ap.error("--out required")

    out = _shard_path(args.out, args.shard)
    jsonl = out.with_suffix(out.suffix + ".jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if jsonl.exists():
        with jsonl.open() as f:
            done = {json.loads(line)["key"] for line in f if line.strip()}
        print(f"resume: {len(done)} keys already done in {jsonl.name}")

    targets = [(s, k) for s, k in _targets(args) if k not in done]
    print(f"{len(targets)} covers to process")

    fails, n_ok, t0 = [], 0, time.time()
    with jsonl.open("a") as sink:
        for i, (src, key) in enumerate(targets, 1):
            try:
                r = extract_one(src, cache_key=key)
                r["key"] = key
                sink.write(json.dumps(r) + "\n")
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                fails.append((key, repr(e)))
                if len(fails) <= 5:
                    traceback.print_exc()
            if i % args.checkpoint == 0 or i == len(targets):
                sink.flush()
                _jsonl_to_parquet(jsonl, out)
                print(f"  {i}/{len(targets)}  {(time.time()-t0)/i:.2f}s/img  "
                      f"{n_ok} ok  {len(fails)} fail  -> {out.name}")

    if fails:
        fp = out.with_suffix(".failures.csv")
        pd.DataFrame(fails, columns=["key", "error"]).to_csv(fp, index=False)
        print(f"{len(fails)} failures -> {fp}")


def _jsonl_to_parquet(jsonl: Path, out: Path) -> None:
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    pd.DataFrame(rows).set_index("key").to_parquet(out)


if __name__ == "__main__":
    main()
