"""Smoke tests for the Streamlit app (run headless with streamlit.testing)."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from motfail.app_data import APP_SAMPLE, load_app_sample

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / "app" / "Home.py"
PAGES = sorted((ROOT / "app" / "pages").glob("*.py"))


def test_five_pages_exist():
    assert [p.name for p in PAGES] == [
        "1_Explore.py",
        "2_Preprocessing.py",
        "3_Train.py",
        "4_Predict.py",
        "5_Evaluate.py",
    ]


def test_home_runs():
    at = AppTest.from_file(str(HOME), default_timeout=60).run()
    assert not at.exception
    assert any(m.label == "Tests in sample" for m in at.metric)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_each_page_runs(page):
    at = AppTest.from_file(str(page), default_timeout=60).run()
    assert not at.exception


def test_home_shows_error_when_sample_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("MOTFAIL_APP_SAMPLE", str(tmp_path / "missing.parquet"))
    at = AppTest.from_file(str(HOME), default_timeout=60).run()
    assert not at.exception
    assert len(at.error) >= 1
    assert "build_dataset.py" in at.error[0].value


def test_load_app_sample_adds_derived_columns():
    df = load_app_sample()
    assert {"fail", "vehicle_age_years", "age_band"} <= set(df.columns)
    assert set(df["fail"].unique()) <= {0, 1}
    assert df["vehicle_age_years"].min() >= 0


def test_load_app_sample_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_dataset.py"):
        load_app_sample(tmp_path / "nope.parquet")


def test_app_sample_within_limits():
    assert APP_SAMPLE.stat().st_size <= 20 * 1024 * 1024
    assert len(pd.read_parquet(APP_SAMPLE, columns=["test_id"])) <= 200_000
