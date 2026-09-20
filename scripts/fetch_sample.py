"""Download a fixed random sample of covers for smoke tests / EDA.

  python scripts/fetch_sample.py            # 50 covers, seed 42 -> data/sample_covers/
  python scripts/fetch_sample.py -n 200 --seed 7 --out data/sample200

Deterministic given (n, seed): the same books every run, so the smoke-test numbers in
build-log.md are reproducible. Covers come from data/bbe_github.csv (see build-log 2026-09-04
for why that file and not the Kaggle mirror).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import scripts  # noqa: F401  (sets SSL_CERT_FILE for the fetch)
from scripts.cover_io import canonical_url, fetch_raw

CSV = "data/bbe_github.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/sample_covers")
    args = ap.parse_args()

    df = pd.read_csv(CSV, low_memory=False)
    df = df[df["coverImg"].astype(str).str.startswith("http")]
    # draw n + 20% headroom then take the first n that download, so a transient fetch
    # failure drops that book rather than shifting the whole sample.
    picks = df.sample(args.n + max(5, args.n // 5), random_state=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for _, row in picks.iterrows():
        dst = out / f"{row['bookId']}.jpg"
        if dst.exists():
            ok += 1
        else:
            try:
                dst.write_bytes(fetch_raw(canonical_url(row["coverImg"]), cache_key=row["bookId"]))
                ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"  skip {row['bookId']}: {e}")
        if ok >= args.n:
            break
    print(f"{ok} covers in {out}/  (n={args.n}, seed={args.seed})")


if __name__ == "__main__":
    main()
