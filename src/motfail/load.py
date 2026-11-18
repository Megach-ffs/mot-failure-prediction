"""Load the committed analysis files and DVSA lookups (one source for notebook and app)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def repo_root() -> Path:
    """The repository root (the package is installed in editable mode: `-e .`)."""
    return Path(__file__).resolve().parents[2]


def processed_path(name: str) -> Path:
    return repo_root() / "data" / "processed" / name


def load_sample(year: int = 2025, columns: list[str] | None = None) -> pd.DataFrame:
    """The analysis sample: one initial car test per vehicle (data/SOURCE.md)."""
    return pd.read_parquet(processed_path(f"mot_{year}_sample.parquet"), columns=columns)


def load_history(year: int = 2025, columns: list[str] | None = None) -> pd.DataFrame:
    """Previous-test features, one row per sample test_id."""
    return pd.read_parquet(processed_path(f"mot_{year}_history.parquet"), columns=columns)


def load_fuel_lookup() -> pd.DataFrame:
    """DVSA fuel-type codes → names (columns fuel_code, fuel_name)."""
    df = pd.read_csv(repo_root() / "data" / "lookup" / "dft_mdr_fuel_type_extract.csv", dtype="string")
    return df.rename(columns={"type_code": "fuel_code", "fuel_type": "fuel_name"})


def load_outcome_lookup() -> pd.DataFrame:
    """DVSA test-outcome codes → names (columns result_code, result_name)."""
    df = pd.read_csv(repo_root() / "data" / "lookup" / "dft_mdr_test_outcome_extract.csv", dtype="string")
    return df.rename(columns={"result": "result_name"})
