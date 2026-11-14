"""Data loading for the Streamlit app: the committed sample plus display columns."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from motfail.data import make_label

# Resolves inside the repo because the package is installed in editable mode (`-e .`).
APP_SAMPLE = Path(__file__).resolve().parents[2] / "data" / "app" / "sample_2025_01.parquet"

AGE_BINS = [0, 3, 6, 9, 12, 15, 20, 200]
AGE_LABELS = ["<3", "3–5", "6–8", "9–11", "12–14", "15–19", "20+"]

# Display grouping of fuel codes. The raw field mixes codes ("HY") and words
# ("HYBRID ELECTRIC (CLEAN)"). Provisional: verify against the DVSA lookup
# tables during week-2 cleaning.
FUEL_GROUPS = {
    "PE": "Petrol",
    "DI": "Diesel",
    "HY": "Hybrid",
    "HYBRID ELECTRIC (CLEAN)": "Hybrid",
    "ED": "Hybrid",
    "EL": "Electric",
    "ELECTRIC": "Electric",
}


def sample_path() -> Path:
    """The app sample path; the MOTFAIL_APP_SAMPLE env var overrides it (used in tests)."""
    return Path(os.environ.get("MOTFAIL_APP_SAMPLE", APP_SAMPLE))


def load_app_sample(path: Path | None = None) -> pd.DataFrame:
    """Load the app sample and add `fail`, `vehicle_age_years`, `age_band`, `fuel_group`."""
    path = Path(path) if path is not None else sample_path()
    if not path.exists():
        raise FileNotFoundError(
            f"App data not found at {path}. Build it with: "
            "uv run python scripts/build_dataset.py --year 2025 --month 1"
        )
    df = pd.read_parquet(path)
    df["fail"] = make_label(df)
    test_date = pd.to_datetime(df["test_date"], errors="coerce")
    first_use = pd.to_datetime(df["first_use_date"], errors="coerce")
    df["vehicle_age_years"] = (test_date - first_use).dt.days / 365.25
    df["age_band"] = pd.cut(
        df["vehicle_age_years"], bins=AGE_BINS, labels=AGE_LABELS, right=False
    )
    df["fuel_group"] = df["fuel_type"].map(FUEL_GROUPS).fillna("Other")
    return df
