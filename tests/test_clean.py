"""Tests for motfail.clean: one test per rule, the report, and the leakage guard."""

import numpy as np
import pandas as pd
import pytest

from motfail import clean as c


def _num(values):
    return pd.Series(values, dtype="float64")


def test_duplicate_tests_are_dropped():
    df = pd.DataFrame({"test_id": [1, 1, 2]})
    out, n = c.drop_duplicate_tests(df)
    assert n == 1 and list(out["test_id"]) == [1, 2]


def test_near_retests_are_dropped_and_no_previous_test_is_kept():
    df = pd.DataFrame({"days_since_prev": _num([5, 29, 30, 365, np.nan])})
    out, n = c.drop_near_retests(df)
    assert n == 2
    assert out["days_since_prev"].tolist()[:2] == [30, 365] and out["days_since_prev"].isna().iloc[2]


def test_dates_are_parsed_and_bad_values_counted():
    df = pd.DataFrame({"test_date": ["2025-01-02", "not a date"], "first_use_date": ["2015-01-01", "2010-05-05"]})
    out, n = c.parse_dates(df)
    assert n == 1
    assert pd.api.types.is_datetime64_any_dtype(out["test_date"]) and out["test_date"].isna().iloc[1]


def test_fuel_names_are_recoded_and_codes_left_alone():
    df = pd.DataFrame({"fuel_type": pd.Categorical(["PE", "ELECTRIC", "HYBRID ELECTRIC (CLEAN)", "EL"])})
    out, n = c.recode_fuel_names(df, {"ELECTRIC": "EL", "HYBRID ELECTRIC (CLEAN)": "HY"})
    assert n == 2 and out["fuel_type"].tolist() == ["PE", "EL", "HY", "EL"]


def test_fuel_lookup_maps_the_names_seen_in_2025():
    mapping = c.fuel_name_to_code()
    assert mapping["ELECTRIC"] == "EL" and mapping["HYBRID ELECTRIC (CLEAN)"] == "HY"


@pytest.mark.parametrize("column", list(c.PLACEHOLDERS))
def test_placeholders_become_missing(column):
    placeholder = c.PLACEHOLDERS[column][0]
    df = pd.DataFrame({column: pd.Categorical(["REAL", placeholder, None])})
    out, n = c.placeholder_to_missing(df, column)
    assert n == 1 and out[column].isna().tolist() == [False, True, True]
    assert placeholder not in out[column].cat.categories


def test_miles_change_cleared_when_either_reading_is_invalid():
    df = pd.DataFrame({
        "test_mileage":     _num([60_000, 0,       40_000, 300_000, 50_000, 70_000]),
        "miles_since_prev": _num([10_000, -50_000, 40_000, 150_000, -5_000, np.nan]),
        "mileage_rollback": pd.array([0, 1, 0, 0, 1, pd.NA], dtype="Int8"),
    })
    # rows: valid · now 0 · before 0 · change > 100k · real rollback · no previous test
    out, n = c.miles_change_invalid(df)
    assert n == 3
    assert out["miles_since_prev"].isna().tolist() == [False, True, True, True, False, True]
    assert out["mileage_rollback"].isna().tolist() == [False, True, True, True, False, True]
    assert out.loc[4, "mileage_rollback"] == 1  # a plausible rollback is kept: it may be clocking


@pytest.mark.parametrize(
    "column, values, rule_step, expected_missing",
    [
        ("test_mileage", [0, 1, 80_000, 600_000], "mileage 0", [True, False, False, False]),
        ("test_mileage", [0, 1, 80_000, 600_000], f"mileage > {c.MILEAGE_MAX:,}", [False, False, False, True]),
        ("cylinder_capacity", [0, 12, 50, 9_600, 49_999, np.nan], f"engine < {c.CC_MIN} cc (incl. 0)",
         [True, True, False, False, False, True]),
        ("cylinder_capacity", [0, 12, 50, 9_600, 49_999, np.nan], f"engine > {c.CC_MAX:,} cc",
         [False, False, False, False, True, True]),
    ],
)
def test_bound_rules(column, values, rule_step, expected_missing):
    rule = next(r for r in c.RULES if r.step == rule_step)
    out, n = rule.apply(pd.DataFrame({column: _num(values)}))
    assert out[column].isna().tolist() == expected_missing
    assert n == sum(expected_missing) - int(np.isnan(values).sum())


def test_first_use_after_test_is_cleared():
    df = pd.DataFrame({
        "test_date": pd.to_datetime(["2025-04-02", "2025-04-02"]),
        "first_use_date": pd.to_datetime(["2025-04-03", "2015-04-03"]),
    })
    out, n = c.first_use_after_test(df)
    assert n == 1 and out["first_use_date"].isna().tolist() == [True, False]


def _frame():
    return pd.DataFrame({
        "test_id": [1, 2, 3],
        "test_date": ["2025-03-01", "2025-03-01", "2025-03-01"],
        "first_use_date": ["2015-01-01", "2025-03-05", "2012-01-01"],
        "fuel_type": pd.Categorical(["PE", "ELECTRIC", "DI"]),
        "make": pd.Categorical(["FORD", "UNCLASSIFIED", "BMW"]),
        "model": pd.Categorical(["FOCUS", "UNCLASSIFIED", "-"]),
        "colour": pd.Categorical(["RED", "BLUE", "NOT STATED"]),
        "postcode_area": pd.Categorical(["LS", "XX", "M"]),
        "test_mileage": _num([80_000, 0, 120_000]),
        "cylinder_capacity": _num([1_600, np.nan, 0]),
        "days_since_prev": _num([360, np.nan, 10]),
        "miles_since_prev": _num([8_000, np.nan, 1_000]),
        "mileage_rollback": pd.array([0, pd.NA, 0], dtype="Int8"),
    })


def test_clean_reports_every_rule_and_leaves_the_input_unchanged():
    raw = _frame()
    before = raw.copy()
    out, report = c.clean(raw)
    pd.testing.assert_frame_equal(raw, before)
    assert report["step"].tolist() == [r.step for r in c.RULES]
    counts = dict(zip(report["step"], report["rows"], strict=True))
    assert counts[f"test < {c.RETEST_DAYS} days after the previous one (a retest in effect)"] == 1
    assert counts["fuel name instead of code"] == 1 and counts["mileage 0"] == 1
    assert counts["first used after the test date"] == 1
    assert out["test_id"].tolist() == [1, 2]  # row 3 was a near retest


def test_clean_on_the_real_sample():
    from motfail.load import load_history, load_sample

    df = load_sample().merge(load_history(), on="test_id", validate="one_to_one")
    out, report = c.clean(df)
    assert out["vehicle_id"].is_unique and out["test_id"].is_unique
    assert (out["days_since_prev"].dropna() >= c.RETEST_DAYS).all()  # history strictly before, no retests
    assert out["fuel_type"].isin(["PE", "DI", "HY", "EL", "ED", "GB", "OT", "LP", "CN", "FC", "GA", "GD"]).all()
    assert not (out["test_mileage"] == 0).any() and not (out["cylinder_capacity"] == 0).any()
    assert len(out) == len(df) - report.loc[report["action"] == "drop row", "rows"].sum()


@pytest.mark.parametrize("column", list(c.NOT_FEATURES))
def test_leakage_guard_rejects_every_forbidden_column(column):
    with pytest.raises(ValueError, match=column):
        c.check_no_leakage(["vehicle_age", column])


def test_leakage_guard_accepts_pre_test_features():
    c.check_no_leakage(["vehicle_age", "test_mileage", "fuel_type", "prev_fail", "days_since_prev"])
