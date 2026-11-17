"""Choosing which tests go into the analysis sample.

Two rules:
1. One test per vehicle, the *earliest* initial test of the year. Rows stay
   independent (no car counted twice), and the car's previous test always lies
   before the sample year, so history features never use sample-year results.
2. A random sample stratified by month, so every month keeps its exact share
   (fail rates vary by month: seasonality).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def first_test_per_vehicle(keys: pd.DataFrame) -> pd.DataFrame:
    """Keep each vehicle's earliest test (ties: lowest test_id).

    `keys` needs columns test_id, vehicle_id, test_date. Sorts an index array
    (np.lexsort) instead of copying the frame, so it works on ~33M rows in
    limited RAM.
    """
    order = np.lexsort(
        (keys["test_id"].to_numpy(), keys["test_date"].to_numpy(), keys["vehicle_id"].to_numpy())
    )
    vehicles = keys["vehicle_id"].to_numpy()[order]
    is_first = np.ones(len(vehicles), dtype=bool)
    is_first[1:] = vehicles[1:] != vehicles[:-1]
    return keys.iloc[order[is_first]].reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int, by: str, seed: int) -> pd.DataFrame:
    """Random sample of about `n` rows where each group of `by` keeps its share.

    Group sizes use largest-remainder rounding, so they add up to exactly
    min(n, len(df)).
    """
    if n >= len(df):
        return df.copy()
    sizes = df[by].value_counts().sort_index()
    quota = sizes / sizes.sum() * n
    alloc = np.floor(quota).astype(int)
    shortfall = n - alloc.sum()
    alloc[(quota - alloc).sort_values(ascending=False).index[:shortfall]] += 1

    rng = np.random.default_rng(seed)
    parts = [
        group.iloc[rng.choice(len(group), size=alloc[key], replace=False)]
        for key, group in df.groupby(by, sort=True)
    ]
    return pd.concat(parts).sort_index()
