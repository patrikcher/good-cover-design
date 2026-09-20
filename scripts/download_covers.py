"""Bulk-download cover images for the working set to data/covers/<key>.jpg.

  python -m scripts.download_covers                 # all of data/books_english.parquet
  python -m scripts.download_covers --limit 500     # first 500 (smoke)
  python -m scripts.download_covers --workers 8     # parallel fetch

Resumable: existing files are skipped. Failures are logged to data/covers/_failures.csv and
retried on the next run (they're just absent files). Polite-ish: one host, small worker pool,
per-request retry with backoff already in cover_io.fetch_raw.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import scripts  # noqa: F401  (SSL_CERT_FILE)
from scripts.cover_io import fetch_raw

BOOKS = "data/books_english.parquet"
OUT = Path("data/covers")


def _one(key: str, url: str) -> tuple[str, str | None]:
    dst = OUT / f"{key}.jpg"
    if dst.exists() and dst.stat().st_size > 0:
        return key, None
    try:
        dst.write_bytes(fetch_raw(url, cache_key=key, cache=False))
        return key, None
    except Exception as e:  # noqa: BLE001
        return key, repr(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_parquet(BOOKS)[["cover_url"]]
    if args.limit:
        df = df.head(args.limit)
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = [(str(k), u) for k, u in df["cover_url"].items()]
    done = have = fail = 0
    fails: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for key, err in ex.map(lambda ku: _one(*ku), jobs):
            done += 1
            if err:
                fail += 1
                fails.append((key, err))
            else:
                have += 1
            if done % 500 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  {have} ok  {fail} fail")

    if fails:
        fp = OUT / "_failures.csv"
        pd.DataFrame(fails, columns=["key", "error"]).to_csv(fp, index=False)
        print(f"{fail} failures -> {fp} (rerun to retry)")
    print(f"covers on disk: {sum(1 for _ in OUT.glob('*.jpg'))}")


if __name__ == "__main__":
    main()
