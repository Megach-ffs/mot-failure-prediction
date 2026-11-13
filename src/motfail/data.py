"""Population filter, fail label and safe file writes for DVSA MOT test results.

Population (design spec §3): cars (test class 4), initial tests (type "NT"),
with a definitive outcome: pass (P), fail (F) or pass after rectification at
the station (PRS). Retests, other vehicle classes and abandoned/aborted tests
(ABR/ABA) are out of scope.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Column order of the DVSA "test result extract" CSVs (2024 and 2025 releases).
RESULT_COLUMNS: list[str] = [
    "test_id",
    "vehicle_id",
    "test_date",
    "test_class_id",
    "test_type",
    "test_result",
    "test_mileage",
    "postcode_area",
    "make",
    "model",
    "colour",
    "fuel_type",
    "cylinder_capacity",
    "first_use_date",
    "completed_date",
]

KEEP_RESULTS: tuple[str, ...] = ("P", "F", "PRS")

CAR_CLASS = 4
INITIAL_TEST = "NT"

# PRS = defects found and fixed during the test: the car *arrived* failing (spec §3).
_LABEL_MAP = {"P": 0, "F": 1, "PRS": 1}


def filter_initial_car_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Keep initial MOT tests of cars with a P/F/PRS outcome; return a copy."""
    test_type = df["test_type"].astype("string").str.strip()
    result = df["test_result"].astype("string").str.strip()
    mask = (
        (df["test_class_id"] == CAR_CLASS)
        & (test_type == INITIAL_TEST)
        & result.isin(KEEP_RESULTS)
    )
    out = df.loc[mask].copy()
    out["test_type"] = test_type[mask]
    out["test_result"] = result[mask]
    return out


def make_label(df: pd.DataFrame) -> pd.Series:
    """Binary target `fail`: 1 for F or PRS, 0 for P. Raises on any other code."""
    result = df["test_result"].astype("string").str.strip()
    unknown = sorted(set(result.dropna()) - set(_LABEL_MAP)) + (["<NA>"] if result.isna().any() else [])
    if unknown:
        raise ValueError(f"Unexpected test_result codes (filter first): {unknown}")
    return result.map(_LABEL_MAP).astype(np.int8).rename("fail")


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write Parquet via a temporary file so a crash never leaves a half-written target."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
