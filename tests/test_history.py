"""Tests for motfail.history: previous-test features without leakage."""

import pandas as pd

from motfail.history import add_previous_test, count_items_per_test


def _sample():
    return pd.DataFrame(
        {
            "test_id": [900, 901, 902],
            "vehicle_id": [1, 2, 3],
            "test_date": pd.to_datetime(["2025-03-10", "2025-06-01", "2025-02-01"]),
            "test_mileage": [60_000.0, 30_000.0, 10_000.0],
        }
    )


def _prior():
    return pd.DataFrame(
        {
            "test_id": [100, 101, 102, 103],
            "vehicle_id": [1, 1, 2, 2],
            "test_date": pd.to_datetime(["2023-03-01", "2024-03-01", "2024-05-20", "2025-06-01"]),
            "test_result": ["P", "F", "PRS", "P"],
            "test_mileage": [40_000.0, 50_000.0, 35_000.0, 30_000.0],
        }
    )


def test_takes_most_recent_strictly_earlier_test():
    out = add_previous_test(_sample(), _prior())
    assert list(out["prev_test_id"].astype("Int64")) == [101, 102, pd.NA]


def test_same_day_test_is_never_used():
    # vehicle 2 has a prior-table row on its own sample date (2025-06-01): must be ignored
    out = add_previous_test(_sample(), _prior()).set_index("vehicle_id")
    assert out.loc[2, "prev_test_id"] == 102


def test_derived_features():
    out = add_previous_test(_sample(), _prior()).set_index("vehicle_id")
    assert list(out["has_prev_test"]) == [1, 1, 0]
    assert out.loc[1, "prev_fail"] == 1 and out.loc[2, "prev_fail"] == 1
    assert pd.isna(out.loc[3, "prev_fail"])
    assert out.loc[1, "miles_since_prev"] == 10_000
    assert out.loc[1, "days_since_prev"] == (pd.Timestamp("2025-03-10") - pd.Timestamp("2024-03-01")).days
    assert out.loc[2, "mileage_rollback"] == 1  # 30k now, 35k before
    assert out.loc[1, "mileage_rollback"] == 0


def test_row_count_and_order_preserved():
    s = _sample()
    out = add_previous_test(s, _prior())
    assert list(out["test_id"]) == list(s["test_id"])


def test_count_items_per_test():
    items = pd.DataFrame(
        {
            "test_id": [101, 101, 101, 102, 102],
            "rfr_type_code": ["F", "A", "A", "P", "A"],
            "dangerous_mark": [None, None, None, "D", None],
        }
    )
    out = count_items_per_test(items).set_index("test_id")
    assert out.loc[101, "prev_items_F"] == 1
    assert out.loc[101, "prev_items_A"] == 2
    assert out.loc[102, "prev_items_P"] == 1
    assert out.loc[101, "prev_dangerous"] == 0 and out.loc[102, "prev_dangerous"] == 1
