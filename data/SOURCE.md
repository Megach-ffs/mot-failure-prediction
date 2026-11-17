# Data source

**Dataset:** Anonymised MOT tests and results: test result extracts
**Publisher:** Driver and Vehicle Standards Agency (DVSA)
**Dataset page:** https://www.data.gov.uk/dataset/c63fca52-ae4c-4b75-bab5-8b4735e1a4c9/anonymised-mot-tests-and-results
**Bulk files:** https://open.data.dvsa.gov.uk/mot-anonymised/index.html
**Licence:** Open Government Licence v3.0. *Contains public sector information licensed under the Open Government Licence v3.0.* https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

**Source files** (all on `https://edh-dvsa-data-gov-uk-files-prod.s3.eu-west-1.amazonaws.com/`; every zip holds 12 monthly CSVs stored uncompressed, so one month can be streamed with HTTP range requests):

| file | use | first downloaded |
|---|---|---|
| `dft_test_result_extracts_2025.zip` (4.8 GB; S3 Last-Modified 2026-06-19) | the tests we predict | 2026-11-13 (Jan), 2026-11-16 (all months) |
| `dft_test_result_extracts_2024.zip` | each sampled car's previous test | 2026-11-17 |
| `dft_test_item_extracts_2024.zip` | defects found at that previous test | 2026-11-17 |
| `lookup.zip`, `dft_mdr_*_extract.csv`, `dft_item_detail_extract.csv` | code meanings (outcomes, fuel types, defect details) | 2026-11-17 |

## Files in this repo
| file | contents |
|---|---|
| `data/raw/results_2025_MM_initial_cars.parquet` | each month of 2025: all initial car tests with a P/F/PRS outcome (82–109 MB each; **git-ignored**, rebuild below) |
| `data/processed/mot_2025_sample.parquet` | **the analysis sample**: 1,000,000 tests, one per vehicle, stratified by month (26.7 MB). The notebook loads this file. |
| `data/app/sample_2025_01.parquet` | 100,000 random January tests (seed 42), used by the app until the app moves to the analysis sample |
| `data/lookup/` | DVSA lookup tables, unchanged |

## Filter and label
- **Population:**
  - cars only (`test_class_id == 4`)
  - initial tests only (`test_type == "NT"`): retests are consequences of a fail, not new cases
  - definitive outcome only (`test_result ∈ {P, F, PRS}`): abandoned or aborted tests (ABR/ABA) are excluded
- **Label `fail`:** 1 if F (fail) or PRS (pass after rectification at the station: the car arrived with failure-level defects), else 0.

## 2025 in numbers
All initial car tests in 2025: **33,210,951**. Overall fail rate (F+PRS): **27.68%**.

| month | tests | fail rate | | month | tests | fail rate |
|---|---|---|---|---|---|---|
| 01 | 2,958,895 | 28.55% | | 07 | 2,784,824 | 27.82% |
| 02 | 2,732,847 | 28.01% | | 08 | 2,603,579 | 27.56% |
| 03 | 3,275,306 | 27.03% | | 09 | 3,238,347 | 26.95% |
| 04 | 2,439,456 | 27.93% | | 10 | 3,205,764 | 28.57% |
| 05 | 2,462,684 | 26.96% | | 11 | 2,712,247 | 28.32% |
| 06 | 2,640,210 | 26.88% | | 12 | 2,156,792 | 27.56% |

## The analysis sample (`scripts/make_sample.py`, seed 42)
1. **One test per vehicle: the earliest initial test of 2025.**
   - 31,707,812 distinct vehicles; 1,503,139 later repeat initial tests were dropped.
   - Why:
     - Rows stay independent: no car is counted twice.
     - The previous test of every sampled car lies before 2025, so history features never use 2025 results.
   - Effect: the sample fail rate is **28.09%**, against 27.68% for all tests. A second initial test in the same year usually follows a fail and a repair (like a retest), and mostly passes, so dropping it raises the fail rate. The gap is largest in October–December, when more cars have had an earlier 2025 test.
2. **1,000,000 tests stratified by month:** each month keeps its exact share of the 31.7M (largest-remainder rounding).
3. `completed_date` is dropped: it is recorded at the end of the test (leakage).

## Data-quality findings
- **Quote escaping.** DVSA escapes a quote inside a quoted field as `\"`, not the CSV-standard `""`. Example: the model `"STREETZONE 50 2T 12\""` in February 2025. A default CSV reader ends the field early and shifts the rest of the row into the wrong columns. We read with `escapechar="\\"`, tested with the real rows (`tests/test_sources.py`). January was unaffected: the same 2,958,895 rows before and after the fix.
- **Fuel codes.** `dft_mdr_fuel_type_extract.csv` confirms `ED` = Electric Diesel (a diesel hybrid) and `HY` = Hybrid Electric (Clean).

## Rebuild
```bash
for m in $(seq 1 12); do uv run python scripts/build_dataset.py --year 2025 --month $m; done   # about 2 min a month
uv run python scripts/make_sample.py --year 2025                                               # about 35 s, about 2.5 GB RAM
```
`build_dataset.py` skips months already built (`--force` rebuilds). Reads time out after 120 s without data and are retried 3 times.
