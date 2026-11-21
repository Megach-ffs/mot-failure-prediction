"""Row-level cleaning of the analysis frame (sample joined to its history).

Every rule here looks at one row at a time and uses fixed, domain-based limits,
never statistics learned from the data. That makes it safe to run before the
train/test split. Anything that is learned from data (imputation values,
scaling) belongs in the modelling pipeline, where it is fitted on training
rows only.

`clean()` runs the rules in order and returns a report with the number of
rows each rule changed, for the notebook and the app.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from motfail.load import load_fuel_lookup

# --- limits (each one is justified in the notebook, section C.1) ---
RETEST_DAYS = 30  # a test this soon after the previous one is a retest in effect
MILEAGE_MAX = 500_000  # above the 99.99th percentile (about 460,000); the values look like typos
MILES_CHANGE_MAX = 100_000  # more than 100,000 miles between two tests at most 2 years apart
CC_MIN, CC_MAX = 50, 10_000  # no car engine under 50 cc; motorhomes on truck chassis reach 9.6 L

# DVSA placeholder values meaning "not known". Postcode area "XX" is NOT one:
# the DVSA user guide (v5.1) says it merges regions with fewer than 5 test
# stations, so it stays a real level.
PLACEHOLDERS = {
    "make": ("UNCLASSIFIED", "UNKNOWN UNKNOWN"),
    "model": ("UNCLASSIFIED", "-", "."),
    "colour": ("NOT STATED",),
}
# DVLA gives vehicles with an unknown date of manufacture this first-use date (user guide v5.1)
UNKNOWN_FIRST_USE = pd.Timestamp("1971-01-01")

# Columns that must never be model features, with the reason.
NOT_FEATURES = {
    "test_id": "identifier",
    "vehicle_id": "identifier (used only to check that no car is on both sides of the split)",
    "test_result": "the target's source",
    "fail": "the target",
    "completed_date": "recorded when the test ends (leakage)",
    "test_class_id": "constant after filtering (cars only)",
    "test_type": "constant after filtering (initial tests only)",
    "test_date": "raw date; used only through derived features (age, month)",
    "first_use_date": "raw date; used only through derived features (age)",
}


def check_no_leakage(features: list[str]) -> None:
    """Raise if any column in `features` is one that must never be a feature."""
    bad = sorted(set(features) & set(NOT_FEATURES))
    if bad:
        raise ValueError(f"not allowed as features: {bad}")


# --- rules: each takes a frame and returns (new frame, rows changed) ---


def drop_duplicate_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    dup = df.duplicated("test_id")
    return df.loc[~dup].reset_index(drop=True), int(dup.sum())


def drop_near_retests(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop tests taken less than RETEST_DAYS after the previous test (NaN = no previous test, kept)."""
    near = df["days_since_prev"] < RETEST_DAYS
    return df.loc[~near].reset_index(drop=True), int(near.sum())


def parse_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Convert the date strings to dates; count values that fail to parse (they become missing)."""
    out = df.copy()
    failed = 0
    for col in ("test_date", "first_use_date"):
        parsed = pd.to_datetime(out[col], errors="coerce")
        failed += int((parsed.isna() & out[col].notna()).sum())
        out[col] = parsed
    return out, failed


def fuel_name_to_code() -> dict[str, str]:
    """Upper-case fuel names from DVSA's lookup → their codes, e.g. 'ELECTRIC' → 'EL'."""
    lookup = load_fuel_lookup()
    return dict(zip(lookup["fuel_name"].str.upper(), lookup["fuel_code"], strict=True))


def recode_fuel_names(
    df: pd.DataFrame, name_to_code: dict[str, str] | None = None
) -> tuple[pd.DataFrame, int]:
    """Some rows hold the fuel's name instead of its code: map them back to the code."""
    name_to_code = fuel_name_to_code() if name_to_code is None else name_to_code
    fuel = df["fuel_type"].astype("string")
    named = fuel.isin(list(name_to_code))
    out = df.copy()
    out["fuel_type"] = fuel.where(~named, fuel.map(name_to_code)).astype("category")
    return out, int(named.sum())


def placeholder_to_missing(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    """DVSA's "not known" values (PLACEHOLDERS) become missing."""
    hit = df[column].isin(PLACEHOLDERS[column])
    out = df.copy()
    out[column] = out[column].mask(hit)
    if isinstance(out[column].dtype, pd.CategoricalDtype):
        out[column] = out[column].cat.remove_unused_categories()
    return out, int(hit.sum())


def miles_change_invalid(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Clear miles_since_prev and mileage_rollback when either reading is invalid.

    Runs before the mileage rules, because it needs the raw readings. The
    previous reading is implied: test_mileage − miles_since_prev.
    """
    now = df["test_mileage"]
    change = df["miles_since_prev"]
    before = now - change

    def bad_reading(m):
        return (m == 0) | (m > MILEAGE_MAX)

    bad = change.notna() & (bad_reading(now) | bad_reading(before) | (change.abs() > MILES_CHANGE_MAX))
    out = df.copy()
    out.loc[bad, "miles_since_prev"] = np.nan
    out.loc[bad, "mileage_rollback"] = pd.NA
    return out, int(bad.sum())


def set_missing_where(
    df: pd.DataFrame, column: str, condition: Callable[[pd.Series], pd.Series]
) -> tuple[pd.DataFrame, int]:
    hit = condition(df[column]).fillna(False).astype(bool)
    out = df.copy()
    out.loc[hit, column] = np.nan
    return out, int(hit.sum())


def first_use_after_test(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """A car cannot be first used after its own test date: clear first_use_date."""
    after = df["first_use_date"] > df["test_date"]
    out = df.copy()
    out.loc[after, "first_use_date"] = pd.NaT
    return out, int(after.sum())


def first_use_placeholder(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """DVLA's stand-in date for an unknown date of manufacture: clear first_use_date."""
    hit = df["first_use_date"] == UNKNOWN_FIRST_USE
    out = df.copy()
    out.loc[hit, "first_use_date"] = pd.NaT
    return out, int(hit.sum())


@dataclass(frozen=True)
class Rule:
    step: str
    column: str
    action: str
    apply: Callable[[pd.DataFrame], tuple[pd.DataFrame, int]]


RULES: list[Rule] = [
    Rule("duplicate test", "test_id", "drop row", drop_duplicate_tests),
    Rule(f"test < {RETEST_DAYS} days after the previous one (a retest in effect)",
         "days_since_prev", "drop row", drop_near_retests),
    Rule("date stored as text", "test_date, first_use_date", "convert to date; unparseable → missing",
         parse_dates),
    Rule("fuel name instead of code", "fuel_type", "recode to the lookup's code", recode_fuel_names),
    *(
        Rule(f"placeholder {', '.join(map(repr, values))}", col, "set missing",
             partial(placeholder_to_missing, column=col))
        for col, values in PLACEHOLDERS.items()
    ),
    Rule(f"invalid reading at either test, or change > {MILES_CHANGE_MAX:,}",
         "miles_since_prev, mileage_rollback", "set missing", miles_change_invalid),
    Rule("mileage 0", "test_mileage", "set missing",
         partial(set_missing_where, column="test_mileage", condition=lambda s: s == 0)),
    Rule(f"mileage > {MILEAGE_MAX:,}", "test_mileage", "set missing",
         partial(set_missing_where, column="test_mileage", condition=lambda s: s > MILEAGE_MAX)),
    Rule(f"engine < {CC_MIN} cc (incl. 0)", "cylinder_capacity", "set missing",
         partial(set_missing_where, column="cylinder_capacity", condition=lambda s: s < CC_MIN)),
    Rule(f"engine > {CC_MAX:,} cc", "cylinder_capacity", "set missing",
         partial(set_missing_where, column="cylinder_capacity", condition=lambda s: s > CC_MAX)),
    Rule("first used after the test date", "first_use_date", "set missing", first_use_after_test),
    Rule("first use 1971-01-01 (DVLA's date for 'unknown')", "first_use_date", "set missing",
         first_use_placeholder),
]


def clean(df: pd.DataFrame, rules: list[Rule] = RULES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply `rules` in order. Returns the cleaned copy and a report (one row per rule)."""
    n_start = len(df)
    rows = []
    for rule in rules:
        df, n = rule.apply(df)
        rows.append({"step": rule.step, "column": rule.column, "action": rule.action,
                     "rows": n, "share_%": round(100 * n / n_start, 3)})
    return df, pd.DataFrame(rows)
