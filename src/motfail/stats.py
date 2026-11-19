"""Association measures for EDA."""

from __future__ import annotations

import math

import pandas as pd
from scipy.stats import chi2_contingency


def cramers_v(x: pd.Series, y: pd.Series, bias_corrected: bool = False) -> float:
    """Cramér's V between two categorical variables (0 = no association, 1 = perfect).

    V = sqrt(chi2 / (n * (min(r, c) - 1))), with chi2 computed without Yates'
    correction (so a 2x2 table gives |phi|).

    With many categories and sparse cells the plain V is biased upwards.
    `bias_corrected=True` applies Bergsma's (2013) correction (beyond the slides).
    """
    table = pd.crosstab(x, y)
    r, c = table.shape
    if min(r, c) < 2:
        return 0.0
    n = table.to_numpy().sum()
    chi2 = chi2_contingency(table, correction=False)[0]
    phi2 = chi2 / n
    if not bias_corrected:
        return math.sqrt(phi2 / (min(r, c) - 1))
    phi2_c = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    r_c = r - (r - 1) ** 2 / (n - 1)
    c_c = c - (c - 1) ** 2 / (n - 1)
    denom = min(r_c - 1, c_c - 1)
    return math.sqrt(phi2_c / denom) if denom > 0 else 0.0
