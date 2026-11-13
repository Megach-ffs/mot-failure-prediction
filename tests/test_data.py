"""Tests for motfail.data: population filter, fail label, atomic Parquet writes."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from motfail.data import (
    KEEP_RESULTS,
    RESULT_COLUMNS,
    filter_initial_car_tests,
    make_label,
    write_parquet_atomic,
)


@pytest.fixture
def raw() -> pd.DataFrame:
    """Eight raw rows covering the awkward cases seen in the DVSA extracts."""
    rows = [
        # test_id, vehicle_id, date, class, type, result, mileage, area, make, model, colour, fuel, cc, first_use, completed
        (1, 101, "2025-01-02", 4, "NT", "P", 52000.0, "BN", "FORD", "FOCUS", "BLUE", "PE", 1596.0, "2015-03-01", "2025-01-02T10:00:00.000Z"),
        (2, 102, "2025-01-02", 4, "NT", "F", 98000.0, "LS", "VAUXHALL", "CORSA", "RED", "PE", 1229.0, "2012-06-11", "2025-01-02T11:00:00.000Z"),
        (3, 103, "2025-01-03", 4, "NT", "PRS", np.nan, "M", "BMW", "320D", "BLACK", "DI", 1995.0, "2014-09-20", "2025-01-03T09:00:00.000Z"),
        (4, 104, "2025-01-03", 4, "RT", "P", 76000.0, "B", "TOYOTA", "YARIS", "WHITE", "HY", 1497.0, "2016-02-02", "2025-01-03T15:00:00.000Z"),
        (5, 105, "2025-01-04", 1, "NT", "P", 0.0, "BN", "HONDA", "CG125", "RED", "PE", 124.0, "1989-03-09", "2025-01-04T12:00:00.000Z"),
        (6, 106, "2025-01-04", 4, "NT", "ABR", 120000.0, "G", "AUDI", "A4", "GREY", "DI", 1968.0, "2010-01-15", "2025-01-04T13:00:00.000Z"),
        (7, 107, "2025-01-05", 4, " NT ", " F ", 64000.0, "CF", "KIA", "PICANTO", "SILVER", "PE", np.nan, "2017-07-07", "2025-01-05T08:00:00.000Z"),
        (8, 108, "2025-01-05", 4, "NT", "ABA", 33000.0, "EH", "MINI", "COOPER", "GREEN", "PE", 1499.0, "2019-05-05", "2025-01-05T16:00:00.000Z"),
    ]
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def test_result_columns_match_dvsa_file_order():
    assert RESULT_COLUMNS == [
        "test_id", "vehicle_id", "test_date", "test_class_id", "test_type", "test_result",
        "test_mileage", "postcode_area", "make", "model", "colour", "fuel_type",
        "cylinder_capacity", "first_use_date", "completed_date",
    ]
    assert KEEP_RESULTS == ("P", "F", "PRS")


def test_filter_drops_unknown_results_and_strips(raw):
    out = filter_initial_car_tests(raw)
    # kept: 1 (P), 2 (F), 3 (PRS), 7 (" F " stripped); dropped: 4 retest, 5 motorbike, 6 ABR, 8 ABA
    assert sorted(out["test_id"]) == [1, 2, 3, 7]
    assert set(out["test_class_id"]) == {4}
    assert set(out["test_type"]) == {"NT"}
    assert set(out["test_result"]) <= set(KEEP_RESULTS)
    assert out.loc[out["test_id"] == 7, "test_result"].item() == "F"


def test_filter_returns_copy(raw):
    out = filter_initial_car_tests(raw)
    out.loc[:, "make"] = "CHANGED"
    assert "CHANGED" not in set(raw["make"])


def test_label_has_no_nan(raw):
    out = filter_initial_car_tests(raw)
    y = make_label(out)
    assert y.name == "fail"
    assert y.dtype == np.int8
    assert not y.isna().any()
    assert set(y.unique()) <= {0, 1}
    by_id = dict(zip(out["test_id"], y))
    assert by_id == {1: 0, 2: 1, 3: 1, 7: 1}  # PRS counts as fail (spec §3)


def test_make_label_rejects_unknown_code(raw):
    with pytest.raises(ValueError, match="ABA"):
        make_label(raw[raw["test_result"] == "ABA"])


def test_write_parquet_atomic_roundtrip(tmp_path: Path, raw):
    target = tmp_path / "out.parquet"
    write_parquet_atomic(raw, target)
    back = pd.read_parquet(target)
    assert back.shape == raw.shape
    assert not target.with_suffix(".tmp").exists()


def test_write_parquet_atomic_leaves_no_partial_file(tmp_path: Path, raw, monkeypatch):
    target = tmp_path / "out.parquet"

    def boom(self, path, *args, **kwargs):
        Path(path).write_bytes(b"partial")  # simulate a half-written file
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        write_parquet_atomic(raw, target)
    assert not target.exists()
    assert not target.with_suffix(".tmp").exists()
