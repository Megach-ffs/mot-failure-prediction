"""Data loading for the Streamlit app: the app sample (built by scripts/make_app_data.py)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from motfail.features import FEATURES

APP_SAMPLE_NAME = "app_sample_2025.parquet"
CLEANING_REPORT_NAME = "cleaning_report.csv"
REBUILD_HINT = "Build it with: uv run python scripts/make_app_data.py"

AGE_BINS = [0, 3, 6, 9, 12, 15, 20, 200]
AGE_LABELS = ["<3", "3–5", "6–8", "9–11", "12–14", "15–19", "20+"]

# Model features plus what the pages need for display and for respecting the split
APP_COLUMNS = ["test_id", "test_date", "fail", "test_result", "split", *FEATURES]


def app_data_dir() -> Path:
    """`data/app`, found from the working directory first (Streamlit Cloud and local runs
    start in the repo root), then from the package location (editable install)."""
    for root in (Path.cwd(), Path(__file__).resolve().parents[2]):
        if (root / "pyproject.toml").exists() and (root / "data").is_dir():
            return root / "data" / "app"
    return Path.cwd() / "data" / "app"


def sample_path() -> Path:
    """The app sample path; the MOTFAIL_APP_SAMPLE env var overrides it (used in tests)."""
    return Path(os.environ.get("MOTFAIL_APP_SAMPLE", app_data_dir() / APP_SAMPLE_NAME))


def load_app_sample(path: Path | str | None = None) -> pd.DataFrame:
    """Load the app sample and add `age_band` for display."""
    path = Path(path) if path is not None else sample_path()
    if not path.exists():
        raise FileNotFoundError(f"App data not found at {path}. {REBUILD_HINT}")
    df = pd.read_parquet(path)
    df["age_band"] = pd.cut(df["vehicle_age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    return df


def load_cleaning_report(path: Path | str | None = None) -> pd.DataFrame:
    """The cleaning report for the full analysis sample (one row per rule)."""
    path = Path(path) if path is not None else app_data_dir() / CLEANING_REPORT_NAME
    if not path.exists():
        raise FileNotFoundError(f"Cleaning report not found at {path}. {REBUILD_HINT}")
    return pd.read_csv(path)
