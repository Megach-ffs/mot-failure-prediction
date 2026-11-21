# MOT failure prediction

[![CI](https://github.com/Megach-ffs/mot-failure-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Megach-ffs/mot-failure-prediction/actions/workflows/ci.yml)

**Live app:** https://mot-failure-prediction.streamlit.app/

Predicting whether a car in Great Britain will **fail its annual MOT test**, using only facts known *before* the test: vehicle age, mileage, make and model, fuel type, engine size, region and the car's previous MOT result.

The project has two audiences:
- **Car buyers and owners:** "How likely is this car to fail, and why?"
- **Road-safety analysts:** "Which vehicles and regions fail most?"

The project is an end-to-end pipeline (data → EDA → preparation → models → evaluation) in a reproducible Jupyter notebook, with an interactive multi-page Streamlit app.

> Status: in development (November 2026).
> - **Done:** data pipeline for all of 2025 (1M-test analysis sample + previous-test history from 2024); notebook sections A (data and joins), B (EDA) and C (cleaning with counts, missing values, leakage guard, features, stratified 70/15/15 split, preprocessing pipeline, feature ablation); app Home, Explore (filters) and Preprocessing pages; CI.
> - **Planned:** models (D), evaluation (E); app Train, Predict and Evaluate pages; Dockerfile.

## Data
DVSA *Anonymised MOT tests and results*: 2025 test result extracts, joined with the 2024 result and test-item (defect) extracts for each car's previous test. How the files were built, the sampling rules and the data-quality findings are in `data/SOURCE.md`.
Source: https://www.data.gov.uk/dataset/c63fca52-ae4c-4b75-bab5-8b4735e1a4c9/anonymised-mot-tests-and-results

Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Setup
Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is installed automatically.

```bash
git clone https://github.com/Megach-ffs/mot-failure-prediction.git
cd mot-failure-prediction
uv sync                                   # exact versions from uv.lock
uv run pytest                             # run the test suite
uv run jupyter lab notebooks/MLDA_CW1.ipynb   # the analysis notebook
uv run streamlit run app/Home.py          # start the app
```

If you prefer pip: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. That installs what the app needs; to run the notebook too, add `pip install jupyterlab`. Run everything from the repository root.

## Licence
Code: MIT (see `LICENSE`). Data: Open Government Licence v3.0.
