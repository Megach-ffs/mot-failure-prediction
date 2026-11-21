"""Explore: fail rates across vehicle segments, with filters."""

import plotly.express as px
import streamlit as st

from motfail.ui import OGL_NOTE, get_sample

st.set_page_config(page_title="Explore · MOT failure", page_icon="🔎", layout="wide")
st.title("🔎 Explore fail rates")
st.caption("100,000 random tests from the cleaned 2025 sample. Rates are raw: segments also differ in age.")

df = get_sample()
MIN_TESTS = 200  # bars with fewer tests are too noisy to compare


def rate_by(view, col, top=None):
    out = view.groupby(col, observed=True)["fail"].agg(fail_rate="mean", tests="size").reset_index()
    if top:
        out = out[out["tests"] >= MIN_TESTS].sort_values("fail_rate").tail(top)
    return out


def bar(data, x, title, label, horizontal=False):
    kw = {"x": "fail_rate", "y": x, "orientation": "h"} if horizontal else {"x": x, "y": "fail_rate"}
    fig = px.bar(data, **kw, text_auto=".0%", hover_data={"tests": ":,"}, title=title,
                 labels={x: label, "fail_rate": "Fail rate"})
    (fig.update_xaxes if horizontal else fig.update_yaxes)(tickformat=".0%", rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)


with st.sidebar:
    st.header("Filters")
    fuels = sorted(df["fuel_group"].dropna().unique())
    chosen_fuels = st.multiselect("Fuel", fuels, default=fuels)
    age_lo, age_hi = st.slider("Vehicle age (years)", 0, 40, (0, 40))
    top_makes = df["make"].value_counts().index[:30].tolist()
    chosen_makes = st.multiselect("Make (30 most common; empty = all)", top_makes)
    areas = sorted(df["postcode_area"].dropna().unique())
    chosen_areas = st.multiselect("Postcode area of the test station (empty = all)", areas)

mask = df["fuel_group"].isin(chosen_fuels) & df["vehicle_age"].between(age_lo, age_hi + 0.999)
if age_hi == 40:  # the top of the slider means "40 and older"
    mask |= df["fuel_group"].isin(chosen_fuels) & (df["vehicle_age"] >= age_lo) & (df["vehicle_age"] > 40)
if chosen_makes:
    mask &= df["make"].isin(chosen_makes)
if chosen_areas:
    mask &= df["postcode_area"].isin(chosen_areas)
view = df[mask]
if len(view) < MIN_TESTS:
    st.warning(f"Only {len(view):,} tests match the filters; widen them to see reliable rates.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Tests in selection", f"{len(view):,}")
c2.metric("Fail rate in selection", f"{view['fail'].mean():.1%}",
          delta=f"{view['fail'].mean() - df['fail'].mean():+.1%} vs all", delta_color="inverse")
c3.metric("Median mileage", f"{view['test_mileage'].median():,.0f}")

left, right = st.columns(2)
with left:
    bar(rate_by(view, "age_band"), "age_band", "By vehicle age", "Age at test (years)")
    prev = rate_by(view, "prev_result")
    prev["prev_result"] = prev["prev_result"].astype(str).map(
        {"none": "no 2024 test", "P": "passed", "F": "failed", "PRS": "PRS"})
    bar(prev, "prev_result", "By last year's result", "2024 MOT result")
with right:
    bar(rate_by(view, "fuel_group").sort_values("fail_rate"), "fuel_group", "By fuel", "Fuel")
    bar(rate_by(view, "make", top=15), "make", f"Makes with the highest fail rate (≥ {MIN_TESTS} tests)",
        "Make", horizontal=True)

areas_rate = rate_by(view, "postcode_area")
areas_rate = areas_rate[areas_rate["tests"] >= MIN_TESTS].sort_values("fail_rate")
if len(areas_rate) >= 2:
    st.subheader("Postcode areas")
    st.dataframe(
        areas_rate.rename(columns={"postcode_area": "area", "fail_rate": "fail rate"})
        .style.format({"fail rate": "{:.1%}", "tests": "{:,}"}),
        hide_index=True, use_container_width=True, height=260,
    )
    st.caption("`XX` merges regions with fewer than 5 test stations (DVSA). "
               "Areas differ in fleet age, so part of the gap is age, not region (notebook B.11).")

st.info("These are raw rates. Electric cars are mostly young and old cars are mostly petrol, so "
        "segments mix effects. The models separate them by using all features together.")
st.caption(OGL_NOTE)
