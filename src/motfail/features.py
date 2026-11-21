"""Feature engineering on the cleaned frame (output of motfail.clean.clean).

Every feature here is computed from one row with fixed formulas, so it is safe
before the split. Anything learned from data (medians, scaling, target means)
lives in motfail.pipeline and is fitted on training rows only.

Every feature is known before the test starts: the car's details, the test
date and station area (from the booking), and last year's public MOT history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from motfail.clean import check_no_leakage

# Fuel codes grouped for modelling. Gas and other rare fuels (589 cars, 0.06%)
# share one group: too few for a level of their own.
FUEL_GROUPS = {"PE": "Petrol", "DI": "Diesel", "HY": "Hybrid", "ED": "Hybrid", "EL": "Electric"}
MIN_AGE_FOR_RATE = 1.0  # years; keeps miles-per-year finite for very young cars
HISTORY_COUNTS = ["prev_items_A", "prev_items_F", "prev_items_M", "prev_items_P", "prev_dangerous"]

# The model's inputs, by how the pipeline treats them.
NUMERIC_WITH_FLAG = ["test_mileage", "cylinder_capacity", "miles_since_prev"]  # missing is informative
NUMERIC = ["vehicle_age", "miles_per_year", "days_since_prev", *HISTORY_COUNTS]
BINARY = ["has_prev_test", "mileage_rollback"]
LOW_CARD = ["fuel_group", "colour", "prev_result", "test_month"]  # one-hot
HIGH_CARD = ["make", "make_model", "postcode_area"]  # target-encoded
FEATURES = NUMERIC_WITH_FLAG + NUMERIC + BINARY + LOW_CARD + HIGH_CARD
check_no_leakage(FEATURES)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the cleaned frame with the engineered features added."""
    out = df.copy()
    age = (out["test_date"] - out["first_use_date"]).dt.days / 365.25
    out["vehicle_age"] = age
    # usage intensity: two cars of the same age can have very different wear
    out["miles_per_year"] = out["test_mileage"] / age.clip(lower=MIN_AGE_FOR_RATE)
    out["test_month"] = out["test_date"].dt.month.astype("string")
    out["fuel_group"] = out["fuel_type"].astype("string").map(FUEL_GROUPS).fillna("Other")
    # model names repeat across makes (262 names, 17.4% of tests), so the key is make + model;
    # an unknown model stays missing (it is informative: C.2)
    make, model = out["make"].astype("string"), out["model"].astype("string")
    out["make_model"] = (make.fillna("?") + " | " + model).where(model.notna())
    # no previous test → no previous items, no known rollback, result "none" (has_prev_test says why)
    out[HISTORY_COUNTS] = out[HISTORY_COUNTS].fillna(0)
    out["mileage_rollback"] = out["mileage_rollback"].astype("Float64").fillna(0).astype("float64")
    out["prev_result"] = out["prev_result"].astype("string").fillna("none")
    for col in ("colour", "postcode_area", "make"):
        out[col] = out[col].astype("string")
    out["has_prev_test"] = out["has_prev_test"].astype("float64")
    return out


def feature_frame(df: pd.DataFrame, columns: list[str] = FEATURES) -> tuple[pd.DataFrame, pd.Series]:
    """X (`columns`; text columns as object dtype with NaN for missing) and y (`fail`)."""
    check_no_leakage(columns)
    X = df[columns].copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
    return X, df["fail"].astype(int)
