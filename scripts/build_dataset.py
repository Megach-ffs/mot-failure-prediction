"""Build a filtered dataset from one month of DVSA MOT test results.

Streams the month's CSV straight out of the yearly zip on S3 (HTTP range
requests; the CSV is stored uncompressed inside the zip), keeps initial car
tests with a P/F/PRS outcome, and writes:

  data/raw/results_<year>_<mm>_initial_cars.parquet   full filtered month (git-ignored)
  data/app/sample_<year>_<mm>.parquet                  uniform random sample for the app

Usage:
  uv run python scripts/build_dataset.py --year 2025 --month 1 [--app-sample-rows 100000] [--seed 42]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from remotezip import RemoteZip

from motfail.data import make_label, write_parquet_atomic
from motfail.sources import iter_filtered_chunks, member_name, results_zip_url

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--app-sample-rows", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    url, member = results_zip_url(args.year), member_name(args.year, args.month)
    print(f"Streaming {member}\n  from {url}")
    t0 = time.time()
    chunks = []
    with RemoteZip(url) as z:
        member_bytes = z.getinfo(member).file_size
        with z.open(member) as f:
            for i, chunk in enumerate(iter_filtered_chunks(f)):
                chunks.append(chunk)
                print(f"  chunk {i + 1}: kept {len(chunk):,} rows ({time.time() - t0:.0f}s)", flush=True)
    df = pd.concat(chunks, ignore_index=True)
    fail = make_label(df)

    tag = f"{args.year}_{args.month:02d}"
    raw_path = ROOT / "data" / "raw" / f"results_{tag}_initial_cars.parquet"
    app_path = ROOT / "data" / "app" / f"sample_{tag}.parquet"
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
