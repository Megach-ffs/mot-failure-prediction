"""Tests for motfail.features and motfail.pipeline: features, split, and the preprocessor."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from motfail import features as f
from motfail.pipeline import make_preprocessor, split


def _clean_frame(n=400, seed=0):
    """A small frame shaped like the output of motfail.clean.clean."""
    rng = np.random.default_rng(seed)
    has_prev = rng.random(n) < 0.9
    makes = rng.choice(["FORD", "BMW", "KIA"], n)
    return pd.DataFrame({
        "test_id": np.arange(n),
        "vehicle_id": np.arange(n) + 10_000,
        "test_date": pd.to_datetime("2025-03-01") + pd.to_timedelta(rng.integers(0, 300, n), "D"),
        "first_use_date": pd.to_datetime("2012-01-01") + pd.to_timedelta(rng.integers(0, 4000, n), "D"),
        "fuel_type": pd.Categorical(rng.choice(["PE", "DI", "HY", "EL", "LP"], n)),
        "make": pd.Categorical(makes),
        "model": pd.Categorical(np.where(rng.random(n) < 0.05, None, rng.choice(["COUPE", "FIESTA"], n))),
        "colour": pd.Categorical(rng.choice(["RED", "BLUE", None], n)),
        "postcode_area": pd.Categorical(rng.choice(["LS", "M", "B"], n)),
        "test_mileage": np.where(rng.random(n) < 0.02, np.nan, rng.integers(5_000, 150_000, n)).astype(float),
        "cylinder_capacity": np.where(rng.random(n) < 0.05, np.nan, rng.integers(900, 3_000, n)).astype(float),
        "has_prev_test": has_prev.astype("int8"),
        "prev_result": pd.Categorical(np.where(has_prev, rng.choice(["P", "F", "PRS"], n), None)),
        "days_since_prev": np.where(has_prev, rng.integers(300, 400, n), np.nan),
        "miles_since_prev": np.where(has_prev, rng.integers(-500, 15_000, n), np.nan),
        "mileage_rollback": pd.array(np.where(has_prev, rng.integers(0, 2, n), None), dtype="Int8"),
        **{c: np.where(has_prev, rng.integers(0, 4, n), np.nan) for c in f.HISTORY_COUNTS},
        "fail": rng.integers(0, 2, n),
    })


def test_add_features_formulas():
    df = pd.DataFrame({
        "test_date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "first_use_date": pd.to_datetime(["2015-01-01", "2024-07-01"]),
        "test_mileage": [100_000.0, 5_000.0],
        "fuel_type": pd.Categorical(["DI", "GB"]),
        "make": pd.Categorical(["FORD", None]),
        "model": pd.Categorical(["COUPE", "COUPE"]),
        "colour": pd.Categorical(["RED", "RED"]),
        "postcode_area": pd.Categorical(["LS", "LS"]),
        "prev_result": pd.Categorical(["F", None]),
        "mileage_rollback": pd.array([1, pd.NA], dtype="Int8"),
        "has_prev_test": pd.array([1, 0], dtype="int8"),
        **{c: [2.0, np.nan] for c in f.HISTORY_COUNTS},
    })
    out = f.add_features(df)
    assert out["vehicle_age"].round(2).tolist() == [10.0, 0.5]
    # the second car is half a year old: the rate uses at least 1 year
    assert out["miles_per_year"].tolist() == pytest.approx([10_000, 5_000], rel=1e-3)
    assert out["test_month"].tolist() == ["1", "1"]
    assert out["fuel_group"].tolist() == ["Diesel", "Other"]
    assert out["make_model"].tolist() == ["FORD | COUPE", "? | COUPE"]  # same model name, different key
    assert out.loc[1, "prev_result"] == "none" and out.loc[1, "mileage_rollback"] == 0
    assert out.loc[1, f.HISTORY_COUNTS].tolist() == [0, 0, 0, 0, 0]


def test_unknown_model_stays_missing():
    df = _clean_frame()
    out = f.add_features(df)
    assert out["make_model"].isna().equals(df["model"].isna())


def test_feature_list_passes_the_leakage_guard_and_rejects_ids():
    f.check_no_leakage(f.FEATURES)
    with pytest.raises(ValueError):
        f.feature_frame(f.add_features(_clean_frame()), [*f.FEATURES, "vehicle_id"])


def test_split_sizes_stratification_and_no_vehicle_overlap():
    df = f.add_features(_clean_frame(n=2_000))
    train, val, test = split(df)
    assert (len(train), len(val), len(test)) == (1_400, 300, 300)
    rates = [p["fail"].mean() for p in (train, val, test)]
    assert max(rates) - min(rates) < 0.01  # stratified
    assert not set(train["vehicle_id"]) & set(test["vehicle_id"])


def test_split_raises_if_a_vehicle_is_in_two_parts():
    df = f.add_features(_clean_frame(n=1_000))
    df["vehicle_id"] = df["vehicle_id"] % 50  # many tests per car
    with pytest.raises(ValueError, match="more than one part"):
        split(df)


def test_preprocessor_output_is_complete_and_fitted_on_train_only():
    df = f.add_features(_clean_frame(n=1_000))
    train, val, _ = split(df)
    X_train, y_train = f.feature_frame(train)
    X_val, _ = f.feature_frame(val)
    X_val.loc[X_val.index[0], "make_model"] = "NEVER | SEEN"  # unseen at prediction time
    pre = make_preprocessor().fit(X_train, y_train)
    Z = pre.transform(X_val)
    assert Z.shape == (len(X_val), len(pre.get_feature_names_out()))
    assert not np.isnan(Z).any()
    names = list(pre.get_feature_names_out())
    assert "num_flag__missingindicator_test_mileage" in names and "high__make_model" in names
    # the imputer's median comes from the training rows only
    imputer = pre.named_transformers_["num_flag"].named_steps["impute"]
    assert imputer.statistics_[0] == X_train["test_mileage"].median()


def test_model_pipeline_fits_and_predicts_probabilities():
    df = f.add_features(_clean_frame(n=1_000))
    train, val, _ = split(df)
    X_train, y_train = f.feature_frame(train)
    X_val, _ = f.feature_frame(val)
    model = Pipeline([("pre", make_preprocessor()), ("clf", LogisticRegression(max_iter=1_000))])
    p = model.fit(X_train, y_train).predict_proba(X_val)[:, 1]
    assert p.shape == (len(X_val),) and ((p >= 0) & (p <= 1)).all()


def test_onehot_variant_and_bad_encoder_name():
    df = f.add_features(_clean_frame(n=1_000))
    X, y = f.feature_frame(df)
    assert make_preprocessor(high_card_encoder="onehot").fit_transform(X, y).shape[0] == len(X)
    with pytest.raises(ValueError):
        make_preprocessor(high_card_encoder="frequency")
