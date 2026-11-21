"""Smoke tests for the Streamlit app (run headless with streamlit.testing)."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from motfail.app_data import (
    APP_COLUMNS,
    load_app_sample,
    load_cleaning_report,
    sample_path,
)

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
    assert "make_app_data.py" in at.error[0].value


def test_load_app_sample_has_features_split_and_age_band():
    df = load_app_sample()
    assert set(APP_COLUMNS) | {"age_band"} <= set(df.columns)
    assert set(df["fail"].unique()) <= {0, 1}
    assert df["vehicle_age"].min() >= 0
    assert set(df["split"].unique()) == {"train", "validation", "test"}


def test_cleaning_report_matches_the_rules():
    from motfail.clean import RULES

    report = load_cleaning_report()
    assert report["step"].tolist() == [r.step for r in RULES]
    assert int(report["rows_before"].iloc[0]) == 1_000_000


def test_load_app_sample_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_app_data.py"):
        load_app_sample(tmp_path / "nope.parquet")


def test_app_sample_within_limits():
    assert sample_path().stat().st_size <= 20 * 1024 * 1024
    assert len(pd.read_parquet(sample_path(), columns=["test_id"])) <= 200_000


def test_home_shows_error_when_sample_corrupt(monkeypatch, tmp_path):
    bad = tmp_path / "corrupt.parquet"
    bad.write_bytes(b"not a parquet file")
    monkeypatch.setenv("MOTFAIL_APP_SAMPLE", str(bad))
    at = AppTest.from_file(str(HOME), default_timeout=60).run()
    assert not at.exception
    assert len(at.error) >= 1


def test_explore_filters_work():
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "1_Explore.py"), default_timeout=60).run()
    at.sidebar.multiselect[1].select("FORD").run()  # make
    at.sidebar.slider[0].set_value((3, 10)).run()  # age
    assert not at.exception
    assert at.metric[0].label == "Tests in selection"
    n = int(at.metric[0].value.replace(",", ""))
    assert 0 < n < 100_000


def test_preprocessing_controls_work():
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "2_Preprocessing.py"), default_timeout=60).run()
    for option in ["z-score (StandardScaler)", "min-max", "log(1 + x)"]:
        at.radio[0].set_value(option).run()
        assert not at.exception
    at.slider[0].set_value(300).run()
    assert not at.exception
    assert at.metric[0].value == "1,000,000"
