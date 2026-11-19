"""Tests for motfail.stats: Cramér's V (plain and bias-corrected)."""

import math

import numpy as np
import pandas as pd
import pytest

from motfail.stats import cramers_v


def _expand(table: list[list[int]]) -> tuple[pd.Series, pd.Series]:
    """Turn a contingency table into two label columns with those counts."""
    a, b = [], []
    for i, row in enumerate(table):
        for j, n in enumerate(row):
            a += [f"r{i}"] * n
            b += [f"c{j}"] * n
    return pd.Series(a), pd.Series(b)


def test_perfect_association_is_one():
    x, y = _expand([[10, 0], [0, 10]])
    assert cramers_v(x, y) == pytest.approx(1.0)


def test_independence_is_zero():
    x, y = _expand([[10, 10], [10, 10]])
    assert cramers_v(x, y) == pytest.approx(0.0)


def test_two_by_two_equals_abs_phi():
    # phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d)) = 250 / sqrt(375000)
    x, y = _expand([[20, 10], [5, 15]])
    assert cramers_v(x, y) == pytest.approx(250 / math.sqrt(375_000))


def test_bias_correction_shrinks_spurious_association():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.integers(0, 300, 2_000))  # many categories, independent of y
    y = pd.Series(rng.integers(0, 2, 2_000))
    plain = cramers_v(x, y)
    corrected = cramers_v(x, y, bias_corrected=True)
    assert plain > 0.2  # sparse cells inflate the plain statistic
    assert corrected < 0.1
