"""Draw the analysis sample from the 12 monthly filtered parquets.

Pass 1 reads only test_id / vehicle_id / test_date (low memory), keeps each
vehicle's earliest initial test of the year, and draws a month-stratified sample.
Pass 2 reads the full columns for the chosen tests only.

Output: data/processed/mot_<year>_sample.parquet (submitted with the CW).
completed_date is dropped: it is recorded at the end of the test (leakage).

Usage:
  uv run python scripts/make_sample.py --year 2025 [--rows 1000000] [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from motfail.data import make_label, write_parquet_atomic
from motfail.sampling import first_test_per_vehicle, stratified_sample

ROOT = Path(__file__).resolve().parents[1]
CATEGORICAL = ["test_type", "test_result", "postcode_area", "make", "model", "colour", "fuel_type"]


def month_path(year: int, month: int) -> Path:
    return ROOT / "data" / "raw" / f"results_{year}_{month:02d}_initial_cars.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = {m: month_path(args.year, m) for m in range(1, 13)}
    missing = [str(p.relative_to(ROOT)) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing month files (run build_dataset.py first): {missing}")

    # Pass 1: keys only, in compact dtypes (about 13 bytes a row instead of ~40),
    # because ~33M rows of int64 + datetime keys plus sort copies exhausted RAM.
    keys, full_fail = [], {}
    for m, p in paths.items():
        k = pd.read_parquet(p, columns=["test_id", "vehicle_id", "test_date", "test_result"])
        full_fail[m] = (len(k), int(k["test_result"].isin(["F", "PRS"]).sum()))
        for c in ("test_id", "vehicle_id"):
            if not k[c].between(0, np.iinfo(np.uint32).max).all():
                raise SystemExit(f"{c} out of uint32 range in month {m}")
        keys.append(
            pd.DataFrame(
                {
                    "test_id": k["test_id"].to_numpy(np.uint32),
                    "vehicle_id": k["vehicle_id"].to_numpy(np.uint32),
                    "test_date": (pd.to_datetime(k["test_date"]) - pd.Timestamp("1970-01-01"))
                    .dt.days.to_numpy(np.int32),
                    "month": np.int8(m),
                }
            )
        )
        del k
    keys = pd.concat(keys, ignore_index=True)
    n_tests = len(keys)
    firsts = first_test_per_vehicle(keys)
    del keys
    chosen = stratified_sample(firsts, n=args.rows, by="month", seed=args.seed)

    # Pass 2: full rows for the chosen tests.
    parts = []
    for m, p in paths.items():
        ids = chosen.loc[chosen["month"] == m, "test_id"].tolist()
        parts.append(pd.read_parquet(p, filters=[("test_id", "in", ids)]))
    df = pd.concat(parts, ignore_index=True).drop(columns="completed_date")
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    assert len(df) == len(chosen) and df["vehicle_id"].is_unique

    out = ROOT / "data" / "processed" / f"mot_{args.year}_sample.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(df, out)

    all_n = sum(n for n, _ in full_fail.values())
    all_f = sum(f for _, f in full_fail.values())
    month = pd.to_datetime(df["test_date"]).dt.month
    print(
        f"Initial car tests in {args.year}: {n_tests:,} (fail rate {all_f / all_n:.4f})\n"
        f"Distinct vehicles (earliest test kept): {len(firsts):,} "
        f"({n_tests - len(firsts):,} later repeat tests dropped)\n"
        f"Sample: {len(df):,} tests, fail rate {make_label(df).mean():.4f}\n"
        f"Per month (sample n / full fail rate / sample fail rate):"
    )
    for m in range(1, 13):
        sm = df[month == m]
        n, f = full_fail[m]
        print(f"  {m:02d}: {len(sm):>7,}  {f / n:.4f}  {make_label(sm).mean():.4f}")
    print(f"{out.relative_to(ROOT)}: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
