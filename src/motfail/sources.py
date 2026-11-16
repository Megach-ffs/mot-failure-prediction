"""Where the DVSA MOT extracts live, and how to read them in filtered chunks.

The yearly zips are large (about 4.8 GB), but each monthly CSV inside is
*stored* (uncompressed). That lets one month be streamed with HTTP range
requests (see scripts/build_dataset.py) without downloading the whole year.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO

import pandas as pd

from motfail.data import RESULT_COLUMNS, filter_initial_car_tests

S3_BASE = "https://edh-dvsa-data-gov-uk-files-prod.s3.eu-west-1.amazonaws.com/"
DATASET_PAGE = (
    "https://www.data.gov.uk/dataset/c63fca52-ae4c-4b75-bab5-8b4735e1a4c9/"
    "anonymised-mot-tests-and-results"
)

RESULT_DTYPES: dict[str, str] = {
    "test_id": "int64",
    "vehicle_id": "int64",
    "test_date": "string",
    "test_class_id": "int64",
    "test_type": "string",
    "test_result": "string",
    "test_mileage": "float64",
    "postcode_area": "string",
    "make": "string",
    "model": "string",
    "colour": "string",
    "fuel_type": "string",
    "cylinder_capacity": "float64",
    "first_use_date": "string",
    "completed_date": "string",
}


def results_zip_url(year: int) -> str:
    """URL of the yearly zip of monthly test-result CSVs."""
    return f"{S3_BASE}dft_test_result_extracts_{year}.zip"


def member_name(year: int, month: int) -> str:
    """Path of one month's CSV inside the yearly zip."""
    return f"dft_test_result_extracts_{year}/dft_test_result_extract_{year}{month:02d}.csv"


def iter_filtered_chunks(fileobj: IO[bytes], chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
    """Read a test-result CSV in chunks, yielding only initial car tests (P/F/PRS)."""
    reader = pd.read_csv(
        fileobj,
        usecols=RESULT_COLUMNS,
        dtype=RESULT_DTYPES,
        chunksize=chunksize,
        # DVSA writes a quote inside a quoted field as \" (e.g. model "STREETZONE 50 2T 12\"").
        # Without this, the field ends early and the rest of the row shifts columns.
        escapechar="\\",
    )
    for chunk in reader:
        yield filter_initial_car_tests(chunk[RESULT_COLUMNS])
