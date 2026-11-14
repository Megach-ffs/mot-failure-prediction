# MOT failure prediction

[![CI](https://github.com/Megach-ffs/mot-failure-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Megach-ffs/mot-failure-prediction/actions/workflows/ci.yml)

**Live app:** https://mot-failure-prediction.streamlit.app/

Predicting whether a car in Great Britain will **fail its annual MOT test**, using only facts known *before* the test: vehicle age, mileage, make and model, fuel type, engine size and region (previous-year test history planned).

The project has two audiences:
- **Car buyers and owners:** "How likely is this car to fail, and why?"
- **Road-safety analysts:** "Which vehicles and regions fail most?"

Planned: an end-to-end pipeline (data → EDA → preparation → models → evaluation), a reproducible Jupyter notebook, and an interactive multi-page Streamlit app.

> Status: in development (November 2026). Live now: data pipeline for January 2025 (100k-test app sample), app Home and Explore pages, CI.

## Data
DVSA *Anonymised MOT tests and results*: 2025 test result extracts (January built so far; joins with 2024 history and test-item extracts planned). Details in `data/SOURCE.md`.
Source: https://www.data.gov.uk/dataset/c63fca52-ae4c-4b75-bab5-8b4735e1a4c9/anonymised-mot-tests-and-results

Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Setup
Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is installed automatically.

```bash
git clone https://github.com/Megach-ffs/mot-failure-prediction.git
cd mot-failure-prediction
uv sync                                   # exact versions from uv.lock
uv run pytest                             # run the test suite
uv run streamlit run app/Home.py          # start the app
```

If you prefer pip: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

## Licence
Code: MIT (see `LICENSE`). Data: Open Government Licence v3.0.
