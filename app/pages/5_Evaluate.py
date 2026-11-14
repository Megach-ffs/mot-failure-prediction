"""Evaluate: placeholder until its week's milestone."""

import streamlit as st

st.set_page_config(page_title="Evaluate · MOT failure", page_icon="📊", layout="wide")
st.title("📊 Evaluate")
st.info("Model comparison on the held-out test set: ROC and precision-recall curves, a threshold explorer with a live confusion matrix, calibration and results by region. Coming in week 3.")
