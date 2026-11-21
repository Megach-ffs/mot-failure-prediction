"""Build the app's data from the committed analysis files.

Writes:
- data/app/app_sample_2025.parquet: 100,000 random rows of the cleaned, feature-engineered
  analysis frame, with a `split` column (train/validation/test, the same split as the notebook)
- data/app/cleaning_report.csv: the cleaning report for the full 1,000,000-row sample

Usage: uv run python scripts/make_app_data.py
"""

from __future__ import annotations

from motfail.app_data import APP_COLUMNS, app_data_dir
from motfail.clean import clean
from motfail.data import make_label, write_parquet_atomic
from motfail.features import add_features
from motfail.load import load_history, load_sample
from motfail.pipeline import SEED, split

N_APP = 100_000


def main() -> None:
    raw = load_sample().merge(load_history(), on="test_id", how="left", validate="one_to_one")
    raw["fail"] = make_label(raw)
    cleaned, report = clean(raw)
    data = add_features(cleaned)
    _train, val, test = split(data)
    data["split"] = "train"
    data.loc[val.index, "split"] = "validation"
    data.loc[test.index, "split"] = "test"

    app = data.sample(N_APP, random_state=SEED)[APP_COLUMNS].reset_index(drop=True)
    for col in ("split", "fuel_group", "make", "make_model", "colour", "postcode_area", "prev_result", "test_month"):
        app[col] = app[col].astype("category")
    out = app_data_dir()
    out.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(app, out / "app_sample_2025.parquet")
    report.assign(rows_before=len(raw)).to_csv(out / "cleaning_report.csv", index=False)
    print(f"app sample: {len(app):,} rows, fail rate {app['fail'].mean():.3%}; "
          f"split {app['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
