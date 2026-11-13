# Data source

**Dataset:** Anonymised MOT tests and results: test result extracts
**Publisher:** Driver and Vehicle Standards Agency (DVSA)
**Dataset page:** https://www.data.gov.uk/dataset/c63fca52-ae4c-4b75-bab5-8b4735e1a4c9/anonymised-mot-tests-and-results
**Bulk files:** https://open.data.dvsa.gov.uk/mot-anonymised/index.html
**2025 results zip:** https://edh-dvsa-data-gov-uk-files-prod.s3.eu-west-1.amazonaws.com/dft_test_result_extracts_2025.zip (4.8 GB; S3 Last-Modified 2026-06-19; 12 monthly CSVs stored uncompressed)
**Licence:** Open Government Licence v3.0. *Contains public sector information licensed under the Open Government Licence v3.0.* https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

## Files in this repo
| file | contents | built |
|---|---|---|
| `data/raw/results_2025_01_initial_cars.parquet` | January 2025: all 2,958,895 initial car tests with a P/F/PRS outcome (97.7 MB; **git-ignored**, rebuild below) | 2026-11-13 |
| `data/app/sample_2025_01.parquet` | uniform random sample of 100,000 rows of the above (seed 42), used by the app | 2026-11-13 |

## Filter and label
- **Population:**
  - cars only (`test_class_id == 4`)
  - initial tests only (`test_type == "NT"`): retests are consequences of a fail, not new cases
  - definitive outcome only (`test_result ∈ {P, F, PRS}`): abandoned or aborted tests (ABR/ABA) are excluded
- **Label `fail`:** 1 if F (fail) or PRS (pass after rectification at the station: the car arrived with failure-level defects), else 0.
- **January 2025:** P 71.45%, F 23.01%, PRS 5.55%. The fail rate (F+PRS) is **28.55%**.

## Rebuild
```bash
uv run python scripts/build_dataset.py --year 2025 --month 1
```
The script streams only the January CSV out of the yearly zip (HTTP range requests, about 420 MB transferred, about 2 minutes).
