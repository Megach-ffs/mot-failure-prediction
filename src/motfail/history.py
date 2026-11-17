"""Previous-test history features.

For each sampled test, find the same vehicle's most recent initial test
*strictly before* the sample test date (never the same day, never later), and
derive features from it. Only information known before the sample test is used.
"""

from __future__ import annotations

import pandas as pd


def add_previous_test(sample: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    """Attach previous-test features to `sample`, keeping its rows and order.

    sample: test_id, vehicle_id, test_date (datetime), test_mileage
    prior:  test_id, vehicle_id, test_date (datetime), test_result, test_mileage
    """
    left = sample.reset_index(drop=True).assign(_row=lambda d: range(len(d)))
    right = prior.rename(
        columns={
            "test_id": "prev_test_id",
            "test_date": "prev_test_date",
            "test_result": "prev_result",
            "test_mileage": "prev_mileage",
        }
    )[["vehicle_id", "prev_test_id", "prev_test_date", "prev_result", "prev_mileage"]]

    merged = pd.merge_asof(
        left.sort_values("test_date"),
        right.sort_values("prev_test_date"),
        left_on="test_date",
        right_on="prev_test_date",
        by="vehicle_id",
        direction="backward",
        allow_exact_matches=False,  # a same-day test is not "previous"
    )
    out = merged.sort_values("_row").drop(columns="_row").reset_index(drop=True)

    has = out["prev_test_id"].notna()
    out["has_prev_test"] = has.astype("int8")
    out["prev_fail"] = out["prev_result"].isin(["F", "PRS"]).astype("Int8").where(has)
    out["days_since_prev"] = (out["test_date"] - out["prev_test_date"]).dt.days
    out["miles_since_prev"] = out["test_mileage"] - out["prev_mileage"]
    out["mileage_rollback"] = (out["miles_since_prev"] < 0).astype("Int8").where(
        out["miles_since_prev"].notna()
    )
    return out


def count_items_per_test(items: pd.DataFrame) -> pd.DataFrame:
    """Per test_id: number of items of each rfr_type_code, and dangerous items.

    Returns columns test_id, prev_items_<code>..., prev_dangerous.
    """
    counts = pd.crosstab(items["test_id"], items["rfr_type_code"]).add_prefix("prev_items_")
    dangerous = items["dangerous_mark"].notna().groupby(items["test_id"]).sum()
    counts["prev_dangerous"] = dangerous.reindex(counts.index).astype("int64")
    counts.columns.name = None
    return counts.reset_index()
