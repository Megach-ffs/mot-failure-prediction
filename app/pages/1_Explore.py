"""Explore: fail rates across vehicle segments."""

import plotly.express as px
import streamlit as st

from motfail.ui import OGL_NOTE, get_sample

st.set_page_config(page_title="Explore · MOT failure", page_icon="🔎", layout="wide")
st.title("🔎 Explore fail rates")

df = get_sample()

fuels = sorted(df["fuel_group"].unique())
chosen = st.multiselect("Fuel type", fuels, default=fuels)
view = df[df["fuel_group"].isin(chosen)]
if view.empty:
    st.warning("Select at least one fuel type.")
    st.stop()

st.metric("Fail rate in selection", f"{view['fail'].mean():.1%}", help=f"{len(view):,} tests")

by_age = (
    view.groupby("age_band", observed=True)["fail"]
    .agg(fail_rate="mean", tests="size")
    .reset_index()
)
fig_age = px.bar(
    by_age, x="age_band", y="fail_rate", text_auto=".0%",
    hover_data={"tests": ":,"},
    labels={"age_band": "Vehicle age at test (years)", "fail_rate": "Fail rate"},
    title="Fail rate by vehicle age",
)
fig_age.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_age, use_container_width=True)

by_fuel = (
    view.groupby("fuel_group")["fail"]
    .agg(fail_rate="mean", tests="size")
    .reset_index()
    .sort_values("fail_rate", ascending=False)
)
fig_fuel = px.bar(
    by_fuel, x="fuel_group", y="fail_rate", text_auto=".0%",
    hover_data={"tests": ":,"},
    labels={"fuel_group": "Fuel", "fail_rate": "Fail rate"},
    title="Fail rate by fuel type",
)
fig_fuel.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_fuel, use_container_width=True)

st.info(
    "Fuel types differ in age too (electric cars are mostly young), so these raw rates "
    "mix two effects. The models in week 3 separate them."
)
st.caption(OGL_NOTE)
