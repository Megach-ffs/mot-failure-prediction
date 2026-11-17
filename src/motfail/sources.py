"""Where the DVSA MOT extracts live, and how to read them in filtered chunks.

The yearly zips are large (about 4.8 GB), but each monthly CSV inside is
*stored* (uncompressed). That lets one month be streamed with HTTP range
requests (see scripts/build_dataset.py) without downloading the whole year.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import IO

import pandas as pd
import requests
import urllib3
from remotezip import RemoteZip

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


ITEM_COLUMNS = ["test_id", "rfr_type_code", "dangerous_mark"]
ITEM_DTYPES: dict[str, str] = {"test_id": "int64", "rfr_type_code": "string", "dangerous_mark": "string"}

TIMEOUT = (15, 120)  # seconds: (connect, read without receiving any data)
NETWORK_ERRORS = (requests.RequestException, urllib3.exceptions.HTTPError, ConnectionError, TimeoutError)


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


def item_zip_url(year: int) -> str:
    """URL of the yearly zip of monthly test-item (defect) CSVs."""
    return f"{S3_BASE}dft_test_item_extracts_{year}.zip"


def item_member_name(year: int, month: int) -> str:
    """Path of one month's test-item CSV inside the yearly zip."""
    return f"dft_test_item_extracts_{year}/dft_test_item_extract_{year}{month:02d}.csv"


def iter_item_chunks(
    fileobj: IO[bytes], test_ids: set[int] | None = None, chunksize: int = 1_000_000
) -> Iterator[pd.DataFrame]:
    """Read a test-item CSV in chunks, optionally keeping only the given test_ids."""
    reader = pd.read_csv(
        fileobj, usecols=ITEM_COLUMNS, dtype=ITEM_DTYPES, chunksize=chunksize, escapechar="\\"
    )
    for chunk in reader:
        chunk = chunk[ITEM_COLUMNS]
        yield chunk if test_ids is None else chunk[chunk["test_id"].isin(test_ids)]


def read_member(
    url: str,
    member: str,
    chunk_reader: Callable[[IO[bytes]], Iterator[pd.DataFrame]],
    retries: int = 3,
) -> tuple[pd.DataFrame, int]:
    """Stream one CSV out of a remote zip through `chunk_reader`.

    Reads time out after TIMEOUT (a stalled connection otherwise hangs forever)
    and the whole member is retried up to `retries` times.
    Returns (rows kept, member size in bytes).
    """
    t0 = time.time()
    for attempt in range(1, retries + 1):
        try:
            chunks = []
            with RemoteZip(url, timeout=TIMEOUT) as z:
                size = z.getinfo(member).file_size
                with z.open(member) as f:
                    for i, chunk in enumerate(chunk_reader(f)):
                        chunks.append(chunk)
                        print(f"  chunk {i + 1}: kept {len(chunk):,} rows ({time.time() - t0:.0f}s)", flush=True)
            return pd.concat(chunks, ignore_index=True), size
        except NETWORK_ERRORS as e:
            print(f"  attempt {attempt}/{retries} failed: {type(e).__name__}: {e}", flush=True)
            if attempt == retries:
                raise
            time.sleep(10 * attempt)
    raise AssertionError("unreachable")
