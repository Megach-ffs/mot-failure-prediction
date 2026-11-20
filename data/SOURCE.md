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
| `dft_item_group_extract.csv` | test-item hierarchy (item and section names for defects) | 2026-11-19 |

## Files in this repo
| file | contents |
|---|---|
| `data/raw/results_2025_MM_initial_cars.parquet` | each month of 2025: all initial car tests with a P/F/PRS outcome (82–109 MB each; **git-ignored**, rebuild below) |
| `data/processed/mot_2025_history.parquet` | previous-test history features for each sample test (12.3 MB), see below |
| `data/processed/mot_2024_prev_failure_reasons.csv` | failure-level items (F, P) at the sampled cars' previous tests, counted per defect code `rfr_id` (1,181 codes, 237,383 tests); decoded with the lookups below |
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

## Previous-test history (`scripts/build_history.py`)
Output: `data/processed/mot_2025_history.parquet` (12.3 MB), one row per sample `test_id`.

**Method.** For each sampled 2025 test, take the same vehicle's most recent 2024 initial car test (P/F/PRS) **strictly before** the 2025 test date. A same-day or later test is never used (tested in `tests/test_history.py`). Then count that test's items from the 2024 test-item extracts.

**Coverage.**
- 949,307 earlier initial tests were found for the sampled vehicles.
- **909,202 of 1,000,000 sampled tests (90.9%) have a previous test.**
- The rest are mostly first MOTs (cars at 3 years) or cars whose previous test was before 2024.

| feature | meaning |
|---|---|
| `has_prev_test` | 1 if a previous 2024 test exists |
| `prev_result`, `prev_fail` | its outcome; `prev_fail` = F or PRS |
| `days_since_prev`, `miles_since_prev` | gap since it |
| `mileage_rollback` | 1 if the mileage went down |
| `prev_items_A/F/M/P` | number of items of each type at it |
| `prev_dangerous` | number of items with a dangerous mark |

**Item type codes.** These are not decoded in the lookup files, so they were checked against outcomes. The table shows the share of items by previous-test result (921k tests):

| result | A | F | M | P |
|---|---|---|---|---|
| P (pass) | 0.908 | 0 | 0.092 | 0 |
| F (fail) | 0.435 | 0.504 | 0.054 | 0.006 |
| PRS | 0.497 | 0 | 0.066 | 0.437 |

Passed tests carry only A and M items, failed tests carry F items, and PRS tests carry P items but no F items. So: **A** = advisory, **M** = minor defect, **F** = failure item, **P** = failure item rectified at the station.

**First look** (sample):
- Fail rate after a 2024 pass: **23.8%**
- after a 2024 fail: **43.0%**
- with no 2024 test: **20.9%**

**Cleaning** (applied by `src/motfail/clean.py`; every rule and its count is in notebook section C.1):
- 374 tests (0.04%) come less than 30 days after the previous test, and 93% of those followed a fail. They are effectively retests across the year boundary, so they are dropped (the same reason as the one-test-per-vehicle rule).
- 939 tests show a mileage change of more than 100,000 miles (from −873,599 to +986,317): odometer errors. For 1,348 tests the previous reading was 0. In both cases `miles_since_prev` and `mileage_rollback` are set to missing (2,473 tests in all).

## Data-quality findings
- **Quote escaping.** DVSA escapes a quote inside a quoted field as `\"`, not the CSV-standard `""`. Example: the model `"STREETZONE 50 2T 12\""` in February 2025. A default CSV reader ends the field early and shifts the rest of the row into the wrong columns. We read with `escapechar="\\"`, tested with the real rows (`tests/test_sources.py`). January was unaffected: the same 2,958,895 rows before and after the fix.
- **Fuel codes.** `dft_mdr_fuel_type_extract.csv` confirms `ED` = Electric Diesel (a diesel hybrid) and `HY` = Hybrid Electric (Clean).
- **Fuel names instead of codes.** 11,818 sample rows hold the name (`ELECTRIC`, `HYBRID ELECTRIC (CLEAN)`) instead of the code; cleaning maps them back with the lookup.
- **Placeholders** meaning "not known": make `UNCLASSIFIED`/`UNKNOWN UNKNOWN`, model `UNCLASSIFIED`/`-`, colour `NOT STATED`, postcode area `XX`. Cleaning sets them to missing.
- **Impossible values:** mileage 0 (1,785) or over 500,000 (79, e.g. 999,999); engine size under 50 cc (50, incl. 30 at 0) or over 10,000 cc (13, e.g. 55,000); one car first used the day after its test. Cleaning sets them to missing. Engine sizes from 8,100 to 9,603 cc are plausibly real (Dodge SRT-10, US motorhomes) and are kept.

## Rebuild
```bash
for m in $(seq 1 12); do uv run python scripts/build_dataset.py --year 2025 --month $m; done   # about 2 min a month
uv run python scripts/make_sample.py --year 2025                                               # about 35 s, about 2.5 GB RAM
uv run python scripts/build_history.py --year 2025                                             # about 45 min, about 1 GB RAM; resumes from data/raw/history/
```
`build_dataset.py` skips months already built (`--force` rebuilds). Downloads use 8 MB HTTP range requests, 4 in flight, each retried up to 5 times; a single long stream was throttled by S3 to about 170 KB/s.
