"""The split and the preprocessing pipeline (everything that learns from data).

The preprocessor is a ColumnTransformer meant to sit inside a Pipeline with
the model, so every learned value (medians, means and standard deviations,
target means) is fitted on training rows only, and again inside each CV fold.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from motfail.features import BINARY, HIGH_CARD, LOW_CARD, NUMERIC, NUMERIC_WITH_FLAG

SEED = 42


def split(df: pd.DataFrame, seed: int = SEED, val: float = 0.15, test: float = 0.15):
    """Stratified train/validation/test split by `fail`. Raises if a vehicle lands in two parts."""
    rest, test_df = train_test_split(df, test_size=test, stratify=df["fail"], random_state=seed)
    train_df, val_df = train_test_split(
        rest, test_size=val / (1 - test), stratify=rest["fail"], random_state=seed
    )
    parts = [set(p["vehicle_id"]) for p in (train_df, val_df, test_df)]
    if parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]:
        raise ValueError("a vehicle appears in more than one part of the split")
    return train_df, val_df, test_df


def make_preprocessor(
    seed: int = SEED,
    scale: bool = True,
    high_card_encoder: str = "target",
    numeric_with_flag: list[str] = NUMERIC_WITH_FLAG,
    numeric: list[str] = NUMERIC,
    binary: list[str] = BINARY,
    low_card: list[str] = LOW_CARD,
    high_card: list[str] = HIGH_CARD,
) -> ColumnTransformer:
    """Impute, encode and (optionally) scale the FEATURES columns.

    - numeric where missing is informative: median + a missing flag
    - other numeric: median
    - binary flags: passed through
    - low-cardinality categoricals: one-hot, missing as its own level, unseen ignored
    - high-cardinality categoricals: target encoding (mean fail rate per level,
      shrunk towards the overall rate for rare levels; cross-fitted on training data)
    - scaling (z-score) of the numeric outputs, for logistic regression and KNN;
      trees are unaffected by it

    The column lists and `high_card_encoder` ("target" or "onehot") can be
    changed for the comparisons in notebook section C.7; empty lists are skipped.
    """
    scaler = [("scale", StandardScaler())] if scale else []
    target_encoder = TargetEncoder(
        target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=seed)
    )
    if high_card_encoder == "target":
        high = Pipeline([("encode", target_encoder), *scaler])
    elif high_card_encoder == "onehot":  # rare levels (< 200 training rows) pooled into one
        high = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200, sparse_output=True)
    else:
        raise ValueError(f"unknown high_card_encoder: {high_card_encoder}")
    blocks = [
        ("num_flag", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), *scaler]),
         numeric_with_flag),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), *scaler]), numeric),
        ("bin", "passthrough", binary),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_card),
        ("high", high, high_card),
    ]
    return ColumnTransformer([b for b in blocks if b[2]], verbose_feature_names_out=True)
