"""Preprocessing: cleaning rules with counts, missing values, scaling, encoding and the split."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from motfail.app_data import load_cleaning_report
from motfail.features import HIGH_CARD, LOW_CARD, NUMERIC, NUMERIC_WITH_FLAG
from motfail.ui import OGL_NOTE, get_sample

st.set_page_config(page_title="Preprocessing · MOT failure", page_icon="🧹", layout="wide")
st.title("🧹 Preprocessing")
st.markdown(
    "Two kinds of step, kept apart so nothing from the test set leaks into training:\n"
    "1. **Cleaning with fixed limits** (no statistics from other rows), run before the split.\n"
    "2. **Steps that learn from data** (medians, scaling, target means), fitted on training rows only, "
    "inside a scikit-learn `Pipeline`."
)
df = get_sample()
train = df[df["split"] == "train"]

tab_clean, tab_missing, tab_scale, tab_encode, tab_split = st.tabs(
    ["1 · Cleaning", "2 · Missing values", "3 · Scaling", "4 · Encoding", "5 · Split"]
)

with tab_clean:
    try:
        report = load_cleaning_report()
    except FileNotFoundError as e:
        st.error(str(e))
    else:
        n0 = int(report["rows_before"].iloc[0])
        dropped = int(report.loc[report["action"] == "drop row", "rows"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows before", f"{n0:,}")
        c2.metric("Rows dropped", f"{dropped:,}")
        c3.metric("Values corrected or set missing", f"{int(report.loc[report['action'] != 'drop row', 'rows'].sum()):,}")
        st.dataframe(report.drop(columns="rows_before"), hide_index=True, use_container_width=True,
                     column_config={"share_%": st.column_config.NumberColumn("share %", format="%.3f")})
        st.markdown(
            "**Why fixed limits, not the IQR rule:** the IQR rule would flag 19,686 mileages and 44,369 "
            "engine sizes that are real (high-mileage cars, big engines). Each fixed limit was checked "
            "against the values it removes (notebook C.1). Postcode area `XX` is kept: DVSA uses it for "
            "merged regions with fewer than 5 test stations, not for unknown."
        )

with tab_missing:
    miss = df[NUMERIC_WITH_FLAG + ["days_since_prev", "make_model", "make", "colour"]].isna()
    table = pd.DataFrame({
        "missing": miss.sum(),
        "missing %": 100 * miss.mean(),
        "fail % if missing": [100 * df.loc[miss[c], "fail"].mean() if miss[c].any() else np.nan for c in miss],
        "fail % if present": [100 * df.loc[~miss[c], "fail"].mean() for c in miss],
    }).sort_values("missing", ascending=False)
    st.dataframe(table.round(1), use_container_width=True)
    st.markdown(
        "- **No previous test** (about 9%): structural, mostly first MOTs at 3 years. `has_prev_test` flags it.\n"
        "- **Engine size**: almost all electric cars (no engine).\n"
        "- **Mileage**: zero or impossible readings; these cars fail far more often, so the pipeline "
        "imputes the **median and adds a missing flag** (`SimpleImputer(add_indicator=True)`).\n"
        "- **Model**: DVSA could not classify it; kept as its own level by the encoder."
    )

with tab_scale:
    feature = st.selectbox("Numeric feature", NUMERIC_WITH_FLAG + NUMERIC, index=0)
    how = st.radio("Transformation", ["raw", "z-score (StandardScaler)", "min-max", "log(1 + x)"], horizontal=True)
    x_train = train[feature].dropna()
    x = df[feature].dropna()
    if how.startswith("z"):
        x = (x - x_train.mean()) / x_train.std(ddof=0)
    elif how == "min-max":
        x = (x - x_train.min()) / (x_train.max() - x_train.min())
    elif how.startswith("log"):
        x = np.log1p(x.clip(lower=0))
    fig = px.histogram(x, nbins=80, title=f"{feature}: {how}", labels={"value": how})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Statistics (mean, sd, min, max) come from **training rows only**, then are applied to every row. "
        "Logistic regression and KNN need comparable scales; trees do not. Min-max lets one extreme value "
        "squeeze everything else towards 0; z-score keeps outliers visible."
    )

with tab_encode:
    st.markdown(
        f"One-hot for few levels ({', '.join(LOW_CARD)}). **Target encoding** for many levels "
        f"({', '.join(HIGH_CARD)}): each level becomes its fail rate, shrunk towards the overall rate "
        "when the level is rare."
    )
    m = st.slider("Smoothing m (how many 'pseudo-tests' at the overall rate are added)", 0, 500, 50, step=10)
    prior = train["fail"].mean()
    stats = train.groupby("make", observed=True)["fail"].agg(tests="size", raw_rate="mean")
    stats["encoded"] = (stats["tests"] * stats["raw_rate"] + m * prior) / (stats["tests"] + m)
    show = pd.concat([stats.nlargest(6, "tests"), stats[stats["tests"].between(3, 30)].nlargest(6, "raw_rate")])
    st.latex(r"\text{encoded} = \frac{n \cdot \bar{y}_{\text{level}} + m \cdot \bar{y}_{\text{all}}}{n + m}")
    st.dataframe(show.round(3), use_container_width=True)
    st.caption(
        f"Overall training fail rate {prior:.3f}. Big makes barely move; rare makes with extreme raw "
        "rates are pulled towards the overall rate. This m-estimate is an illustration; the model uses "
        "scikit-learn's `TargetEncoder`, which sets the shrinkage from the data and **cross-fits** on "
        "training rows (each row is encoded from the other folds), so a row's own label never leaks "
        "into its feature."
    )

with tab_split:
    st.markdown(
        "**70 / 15 / 15, stratified by fail, seed 42.** Train: fitting and cross-validation. "
        "Validation: threshold and model choice. Test: used once, at the end. Each car appears once, "
        "so no car is on both sides."
    )
    parts = df.groupby("split", observed=True)["fail"].agg(tests="size", fail_rate="mean")
    st.dataframe(parts.style.format({"tests": "{:,}", "fail_rate": "{:.1%}"}), use_container_width=True)
    st.caption("Counts are for this app's 100,000-row sample; the full split is 699,738 / 149,944 / 149,944.")

st.caption(OGL_NOTE)
