"""Tests for motfail.load: the committed analysis files and lookups load with the expected keys."""

from motfail.load import (
    load_fuel_lookup,
    load_history,
    load_outcome_lookup,
    load_sample,
    repo_root,
)


def test_repo_root_contains_pyproject():
    assert (repo_root() / "pyproject.toml").exists()


def test_sample_and_history_share_a_unique_test_id():
    s = load_sample(columns=["test_id", "vehicle_id"])
    h = load_history(columns=["test_id", "has_prev_test"])
    assert s["test_id"].is_unique and h["test_id"].is_unique
    assert s["vehicle_id"].is_unique  # one test per vehicle (D6)
    assert set(s["test_id"]) == set(h["test_id"])


def test_sample_has_no_end_of_test_columns():
    assert "completed_date" not in load_sample().columns  # leakage guard


def test_lookups_have_unique_codes():
    fuel = load_fuel_lookup()
    outcome = load_outcome_lookup()
    assert list(fuel.columns) == ["fuel_code", "fuel_name"] and fuel["fuel_code"].is_unique
    assert list(outcome.columns) == ["result_code", "result_name"] and outcome["result_code"].is_unique
    assert {"P", "F", "PRS"} <= set(outcome["result_code"])
