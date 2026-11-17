"""Tests for motfail.sampling: one test per vehicle, month-stratified sample."""

import pandas as pd

from motfail.sampling import first_test_per_vehicle, stratified_sample


def _keys():
    return pd.DataFrame(
        {
            "test_id": [10, 11, 12, 13, 14, 15],
            "vehicle_id": [1, 1, 2, 3, 3, 3],
            "test_date": pd.to_datetime(
                ["2025-05-01", "2025-02-01", "2025-01-10", "2025-07-01", "2025-07-01", "2025-03-03"]
            ),
        }
    )


def test_first_test_per_vehicle_keeps_the_earliest():
    out = first_test_per_vehicle(_keys())
    assert out["vehicle_id"].is_unique
    assert dict(zip(out["vehicle_id"], out["test_id"])) == {1: 11, 2: 12, 3: 15}


def test_first_test_per_vehicle_tie_breaks_on_test_id():
    keys = _keys()
    keys = keys[keys["test_id"].isin([13, 14])]  # same vehicle, same date
    assert list(first_test_per_vehicle(keys)["test_id"]) == [13]


def test_stratified_sample_keeps_month_shares_and_is_reproducible():
    months = [1] * 600 + [2] * 300 + [3] * 100
    df = pd.DataFrame({"test_id": range(1000), "month": months})
    a = stratified_sample(df, n=100, by="month", seed=42)
    b = stratified_sample(df, n=100, by="month", seed=42)
    assert len(a) == 100
    assert a["month"].value_counts().to_dict() == {1: 60, 2: 30, 3: 10}
    assert a["test_id"].tolist() == b["test_id"].tolist()


def test_stratified_sample_returns_everything_when_n_is_too_big():
    df = pd.DataFrame({"test_id": range(10), "month": [1] * 10})
    assert len(stratified_sample(df, n=50, by="month", seed=0)) == 10
