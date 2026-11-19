"""Previous-test history for the analysis sample.

Stage 1: stream the 2024 results (12 months), keep initial car tests (P/F/PRS)
         of vehicles in the sample.
Stage 2: for each sampled test, take that vehicle's most recent 2024 initial
         test strictly before it (motfail.history.add_previous_test).
Stage 3: stream the 2024 test items (12 months), keep items of those previous
         tests, count them by type.

Per-month downloads are cached in data/raw/history/ (git-ignored), so a rerun
resumes where it stopped.

Outputs:
  data/processed/mot_<year>_history.parquet               one row per sample test_id
  data/processed/mot_<year-1>_prev_failure_reasons.csv    failure-level items of those previous tests, per reason

Usage:
  uv run python scripts/build_history.py --year 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from motfail.data import write_parquet_atomic
from motfail.history import add_previous_test, count_items_per_test
from motfail.sources import (
    item_member_name,
    item_zip_url,
    iter_filtered_chunks,
    iter_item_chunks,
    member_name,
    read_member,
    results_zip_url,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "history"
PRIOR_COLUMNS = ["test_id", "vehicle_id", "test_date", "test_result", "test_mileage"]


def cached(path: Path, build) -> pd.DataFrame:
    if path.exists():
        print(f"  cached: {path.relative_to(ROOT)}")
        return pd.read_parquet(path)
    df = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(df, path)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--year", type=int, required=True, help="the sample year; history comes from year - 1")
    args = ap.parse_args()
    prev_year = args.year - 1

    sample = pd.read_parquet(
        ROOT / "data" / "processed" / f"mot_{args.year}_sample.parquet",
        columns=["test_id", "vehicle_id", "test_date", "test_mileage"],
    )
    sample["test_date"] = pd.to_datetime(sample["test_date"])
    vehicles = set(sample["vehicle_id"])
    print(f"Sample: {len(sample):,} tests, {len(vehicles):,} vehicles")

    # Stage 1: prior results for sampled vehicles.
    prior_parts = []
    for m in range(1, 13):
        member = member_name(prev_year, m)
        print(f"Results {prev_year}-{m:02d}")

        def reader(f):
            for c in iter_filtered_chunks(f):
                yield c.loc[c["vehicle_id"].isin(vehicles), PRIOR_COLUMNS]

        prior_parts.append(
            cached(
                CACHE / f"prior_results_{prev_year}_{m:02d}.parquet",
                lambda member=member, reader=reader: read_member(results_zip_url(prev_year), member, reader)[0],
            )
        )
    prior = pd.concat(prior_parts, ignore_index=True)
    prior["test_date"] = pd.to_datetime(prior["test_date"])
    print(f"Prior initial tests of sampled vehicles: {len(prior):,}")

    # Stage 2: most recent strictly-earlier test.
    hist = add_previous_test(sample, prior)
    prev_ids = set(hist["prev_test_id"].dropna().astype("int64"))
    print(f"Sample tests with a previous test: {len(prev_ids):,} ({hist['has_prev_test'].mean():.4f})")

    # Stage 3: items of those previous tests.
    item_parts = []
    for m in range(1, 13):
        member = item_member_name(prev_year, m)
        print(f"Items {prev_year}-{m:02d}")
        item_parts.append(
            cached(
                CACHE / f"prior_items_{prev_year}_{m:02d}.parquet",
                lambda member=member: read_member(
                    item_zip_url(prev_year), member, lambda f: iter_item_chunks(f, prev_ids)
                )[0],
            )
        )
    items = pd.concat(item_parts, ignore_index=True)
    items = items[items["test_id"].isin(prev_ids)]
    counts = count_items_per_test(items).rename(columns={"test_id": "prev_test_id"})

    # Evidence for what the type codes mean: item mix by previous-test result.
    mix = items.merge(prior[["test_id", "test_result"]], on="test_id")
    print("Item type share by previous-test result (rows: result, columns: rfr_type_code):")
    print(pd.crosstab(mix["test_result"], mix["rfr_type_code"], normalize="index").round(3))

    hist["prev_test_id"] = hist["prev_test_id"].astype("Int64")
    counts["prev_test_id"] = counts["prev_test_id"].astype("Int64")
    out = hist.merge(counts, on="prev_test_id", how="left")
    count_cols = [c for c in counts.columns if c != "prev_test_id"]
    has = out["has_prev_test"] == 1
    out.loc[has, count_cols] = out.loc[has, count_cols].fillna(0)  # a previous test with no items
    out["prev_result"] = out["prev_result"].astype("category")

    keep = [
        "test_id",
        "has_prev_test",
        "prev_result",
        "prev_fail",
        "days_since_prev",
        "miles_since_prev",
        "mileage_rollback",
        *count_cols,
    ]
    out = out[keep]
    assert len(out) == len(sample) and out["test_id"].is_unique

    # Failure-level items (F = failed, P = rectified at the station) of the previous tests,
    # aggregated per reason: the EDA's "top failure reasons" (decoded with data/lookup/).
    fail_items = items[items["rfr_type_code"].isin(["F", "P"])]
    reasons = (
        fail_items.groupby("rfr_id")
        .agg(n_items=("test_id", "size"), n_tests=("test_id", "nunique"))
        .reset_index()
        .sort_values("n_tests", ascending=False)
    )
    reasons_path = ROOT / "data" / "processed" / f"mot_{prev_year}_prev_failure_reasons.csv"
    reasons.to_csv(reasons_path, index=False)
    print(f"{reasons_path.relative_to(ROOT)}: {len(reasons):,} reasons, "
          f"{fail_items['test_id'].nunique():,} previous tests with a failure-level item")

    path = ROOT / "data" / "processed" / f"mot_{args.year}_history.parquet"
    write_parquet_atomic(out, path)
    print(out.describe(include="all").T.to_string())
    print(f"{path.relative_to(ROOT)}: {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
