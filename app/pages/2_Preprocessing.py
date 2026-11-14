"""Preprocessing: placeholder until its week's milestone."""

import streamlit as st

st.set_page_config(page_title="Preprocessing · MOT failure", page_icon="🧹", layout="wide")
st.title("🧹 Preprocessing")
st.info("How the raw DVSA records are cleaned and turned into model features: error corrections (impossible mileage, odometer roll-backs, inconsistent fuel codes), missing values, encoding and scaling, each with before/after counts. Coming in week 2.")
