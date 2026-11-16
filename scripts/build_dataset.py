"""Build a filtered dataset from one month of DVSA MOT test results.

Streams the month's CSV straight out of the yearly zip on S3 (HTTP range
requests; the CSV is stored uncompressed inside the zip), keeps initial car
tests with a P/F/PRS outcome, and writes:

  data/raw/results_<year>_<mm>_initial_cars.parquet   full filtered month (git-ignored)
  data/app/sample_<year>_<mm>.parquet                  uniform random sample for the app

A month whose raw parquet already exists is skipped unless --force is given.
Network reads time out (a stalled connection otherwise hangs forever) and the
whole month is retried up to --retries times.

Usage:
  uv run python scripts/build_dataset.py --year 2025 --month 1 [--app-sample-rows 100000] [--seed 42] [--force]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from remotezip import RemoteZip

from motfail.data import make_label, write_parquet_atomic
from motfail.sources import iter_filtered_chunks, member_name, results_zip_url

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = (15, 120)  # seconds: (connect, read without receiving any data)
NETWORK_ERRORS = (requests.RequestException, ConnectionError, TimeoutError)


def stream_month(url: str, member: str, t0: float) -> tuple[pd.DataFrame, int]:
    """Download one monthly CSV in filtered chunks; return (rows, member size in bytes)."""
    chunks = []
    with RemoteZip(url, timeout=TIMEOUT) as z:
        member_bytes = z.getinfo(member).file_size
        with z.open(member) as f:
            for i, chunk in enumerate(iter_filtered_chunks(f)):
                chunks.append(chunk)
                print(f"  chunk {i + 1}: kept {len(chunk):,} rows ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(chunks, ignore_index=True), member_bytes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--app-sample-rows", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="rebuild even if the output exists")
    args = ap.parse_args()

    tag = f"{args.year}_{args.month:02d}"
    raw_path = ROOT / "data" / "raw" / f"results_{tag}_initial_cars.parquet"
    app_path = ROOT / "data" / "app" / f"sample_{tag}.parquet"
    if raw_path.exists() and not args.force:
        print(f"{raw_path.relative_to(ROOT)} exists: skipping (use --force to rebuild)")
        return

    url, member = results_zip_url(args.year), member_name(args.year, args.month)
    print(f"Streaming {member}\n  from {url}")
    t0 = time.time()
    for attempt in range(1, args.retries + 1):
        try:
            df, member_bytes = stream_month(url, member, t0)
            break
        except NETWORK_ERRORS as e:
            print(f"  attempt {attempt}/{args.retries} failed: {type(e).__name__}: {e}", flush=True)
            if attempt == args.retries:
                raise
            time.sleep(10 * attempt)
    fail = make_label(df)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    app_path.parent.mkdir(parents=True, exist_ok=True)

    write_parquet_atomic(df, raw_path)
    n = min(args.app_sample_rows, len(df))
    write_parquet_atomic(df.sample(n=n, random_state=args.seed), app_path)

    print(
        f"\nSource member size: {member_bytes / 1e6:.0f} MB\n"
        f"Initial car tests kept: {len(df):,}\n"
        f"Outcome shares: {df['test_result'].value_counts(normalize=True).round(4).to_dict()}\n"
        f"Fail rate (F+PRS): {fail.mean():.4f}\n"
        f"{raw_path.relative_to(ROOT)}: {raw_path.stat().st_size / 1e6:.1f} MB\n"
        f"{app_path.relative_to(ROOT)}: {n:,} rows, {app_path.stat().st_size / 1e6:.1f} MB\n"
        f"Done in {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
