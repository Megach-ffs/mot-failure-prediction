"""Where the DVSA MOT extracts live, and how to read them in filtered chunks.

The yearly zips are large (about 4.8 GB), but each monthly CSV inside is
*stored* (uncompressed). That lets one month be streamed with HTTP range
requests (see scripts/build_dataset.py) without downloading the whole year.
"""

from __future__ import annotations

import io
import struct
import time
import zipfile
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
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


PART_SIZE = 8 * 2**20  # bytes per HTTP range request
PART_DEADLINE = 90  # seconds: a part slower than this is abandoned and fetched again

Fetch = Callable[[int, int], bytes]


def http_range_fetcher(url: str, retries: int = 5) -> Fetch:
    """Return fetch(start, n) -> exactly n bytes of `url` from offset `start`.

    Every call is a fresh HTTP range request with its own retries. One long
    streaming download was throttled by S3 to about 170 KB/s while new requests
    got about 10 MB/s, and a trickle never trips a read timeout, so parts that
    run past PART_DEADLINE are abandoned and retried too.
    """

    def fetch(start: int, n: int) -> bytes:
        for attempt in range(1, retries + 1):
            try:
                t0 = time.time()
                headers = {"Range": f"bytes={start}-{start + n - 1}"}
                with requests.get(url, headers=headers, timeout=TIMEOUT, stream=True) as r:
                    if r.status_code != 206:
                        raise OSError(f"expected HTTP 206, got {r.status_code}")
                    data = bytearray()
                    for piece in r.iter_content(2**20):
                        data += piece
                        if time.time() - t0 > PART_DEADLINE:
                            raise TimeoutError(f"part slower than {PART_DEADLINE}s")
                if len(data) != n:
                    raise OSError(f"expected {n} bytes, got {len(data)}")
                return bytes(data)
            except (*NETWORK_ERRORS, OSError) as e:
                print(f"  part at {start:,}: attempt {attempt}/{retries} failed: {type(e).__name__}: {e}", flush=True)
                if attempt == retries:
                    raise
                time.sleep(2 * attempt)
        raise AssertionError("unreachable")

    return fetch


def member_data_range(zf: zipfile.ZipFile, member: str, fetch: Fetch) -> tuple[int, int]:
    """(offset, length) of a *stored* member's bytes inside the zip.

    The data starts after the member's local header: 30 fixed bytes, then the
    file name and an extra field whose lengths are in bytes 26-29.
    """
    info = zf.getinfo(member)
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{member} is compressed; only stored members can be range-read")
    header = fetch(info.header_offset, 30)
    if header[:4] != b"PK\x03\x04":
        raise ValueError(f"no local file header for {member}")
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    return info.header_offset + 30 + name_len + extra_len, info.file_size


class RangeReader(io.RawIOBase):
    """Read-only file object over bytes [start, start + length), fetched in parts.

    Up to `prefetch` parts download in background threads while the caller
    parses the current one; parts are always returned in order.
    """

    def __init__(
        self, fetch: Fetch, start: int, length: int, part_size: int = PART_SIZE, prefetch: int = 4
    ):
        self._fetch, self._start, self._length, self._part = fetch, start, length, part_size
        self._next_offset = 0
        self._pool = ThreadPoolExecutor(max_workers=max(1, prefetch))
        self._queue: deque[Future[bytes]] = deque()
        self._prefetch = max(1, prefetch)
        self._buf, self._bufpos = b"", 0

    def _fill_queue(self) -> None:
        while len(self._queue) < self._prefetch and self._next_offset < self._length:
            n = min(self._part, self._length - self._next_offset)
            self._queue.append(self._pool.submit(self._fetch, self._start + self._next_offset, n))
            self._next_offset += n

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        if self._bufpos >= len(self._buf):
            self._fill_queue()
            if not self._queue:
                return 0
            self._buf, self._bufpos = self._queue.popleft().result(), 0
            self._fill_queue()
        k = min(len(b), len(self._buf) - self._bufpos)
        b[:k] = self._buf[self._bufpos : self._bufpos + k]
        self._bufpos += k
        return k

    def close(self) -> None:
        for fut in self._queue:
            fut.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)
        super().close()


def read_member(
    url: str,
    member: str,
    chunk_reader: Callable[[IO[bytes]], Iterator[pd.DataFrame]],
    retries: int = 5,
) -> tuple[pd.DataFrame, int]:
    """Stream one stored CSV out of a remote zip through `chunk_reader`.

    The zip's central directory is read once (remotezip); the member's bytes are
    then fetched in PART_SIZE range requests, each retried on its own.
    Returns (rows kept, member size in bytes).
    """
    t0 = time.time()
    fetch = http_range_fetcher(url, retries=retries)
    with RemoteZip(url, timeout=TIMEOUT) as z:
        start, length = member_data_range(z, member, fetch)
    chunks = []
    with io.BufferedReader(RangeReader(fetch, start, length), buffer_size=2**20) as f:
        for i, chunk in enumerate(chunk_reader(f)):
            chunks.append(chunk)
            print(f"  chunk {i + 1}: kept {len(chunk):,} rows ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(chunks, ignore_index=True), length
