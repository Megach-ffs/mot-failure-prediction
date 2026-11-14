"""Shared helpers for app pages: cached data loading with a readable error."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from motfail.app_data import load_app_sample, sample_path

REPO_URL = "https://github.com/Megach-ffs/mot-failure-prediction"
OGL_NOTE = (
    "Data: DVSA *Anonymised MOT tests and results*. Contains public sector information "
    "licensed under the [Open Government Licence v3.0]"
    "(https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)."
)


@st.cache_data(show_spinner="Loading MOT sample…")
def _cached_sample(path: str) -> pd.DataFrame:
    return load_app_sample(path)


def get_sample() -> pd.DataFrame:
    """Return the app sample, or show an error and stop the page if it is missing."""
    try:
        return _cached_sample(str(sample_path()))
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:  # noqa: BLE001 — last line of defence: never show users a traceback
        st.error(f"The app data could not be loaded ({type(e).__name__}: {e}).")
        st.stop()
