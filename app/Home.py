"""MOT failure prediction: home page."""

import pandas as pd
import streamlit as st

from motfail.ui import OGL_NOTE, REPO_URL, get_sample

st.set_page_config(page_title="MOT failure prediction", page_icon="🚗", layout="wide")

st.title("🚗 Will this car fail its MOT?")
st.markdown(
    "Every car in Great Britain over three years old needs an annual **MOT** roadworthiness "
    "test. This app explores a 100,000-test sample of real 2025 MOT results and, once the "
    "models are trained, will predict how likely a car is to **fail**, using only facts known "
    "*before* the test (age, mileage, make and model, fuel, engine size, region; previous-test "
    "history is planned)."
)

df = get_sample()

c1, c2, c3 = st.columns(3)
c1.metric("Tests in sample", f"{len(df):,}")
c2.metric("Fail rate (F + PRS)", f"{df['fail'].mean():.1%}")
c3.metric("Median vehicle age", f"{df['vehicle_age_years'].median():.1f} years")

st.markdown(
    """
**What counts as a fail?** A car fails if the tester found failure-level defects: either it
left with a fail (**F**), or the defects were fixed during the test (**PRS**, pass after
rectification). Both mean the car *arrived* in a failing state.

**Pages**
- **Explore**: fail rates by age, fuel and more *(first charts live)*
- **Preprocessing**: cleaning and feature choices *(week 2)*
- **Train**: retrain a model live with your own settings *(week 3)*
- **Predict**: your car's fail probability and main risk factors *(week 3)*
- **Evaluate**: model comparison, ROC/PR curves, threshold explorer *(week 3)*
"""
)
period = pd.to_datetime(df["test_date"])
st.caption(
    f"Sample: {len(df):,} initial car MOT tests, {period.min():%d %b %Y} – {period.max():%d %b %Y}. "
    f"{OGL_NOTE} · [Source code]({REPO_URL})"
)
