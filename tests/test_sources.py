"""Tests for motfail.sources: DVSA URLs and chunked, filtered CSV reading."""

import io

import pandas as pd

from motfail.sources import iter_filtered_chunks, member_name, results_zip_url

CSV = b"""test_id,vehicle_id,test_date,test_class_id,test_type,test_result,test_mileage,postcode_area,make,model,colour,fuel_type,cylinder_capacity,first_use_date,completed_date
1,101,2025-01-02,4,NT,P,52000,BN,FORD,FOCUS,BLUE,PE,1596,2015-03-01,2025-01-02T10:00:00.000Z
2,102,2025-01-02,4,NT,F,98000,LS,VAUXHALL,CORSA,RED,PE,1229,2012-06-11,2025-01-02T11:00:00.000Z
3,103,2025-01-03,4,RT,P,76000,B,TOYOTA,YARIS,WHITE,HY,1497,2016-02-02,2025-01-03T15:00:00.000Z
4,104,2025-01-04,1,NT,P,0,BN,HONDA,CG125,RED,PE,124,1989-03-09,2025-01-04T12:00:00.000Z
5,105,2025-01-04,4,NT,PRS,,M,BMW,320D,BLACK,DI,,2014-09-20,2025-01-04T13:00:00.000Z
6,106,2025-01-05,4,NT,ABA,33000,EH,MINI,COOPER,GREEN,PE,1499,2019-05-05,2025-01-05T16:00:00.000Z
"""


def test_urls_and_member_names():
    assert results_zip_url(2025) == (
        "https://edh-dvsa-data-gov-uk-files-prod.s3.eu-west-1.amazonaws.com/"
        "dft_test_result_extracts_2025.zip"
    )
    assert member_name(2025, 1) == (
        "dft_test_result_extracts_2025/dft_test_result_extract_202501.csv"
    )
    assert member_name(2024, 12).endswith("dft_test_result_extract_202412.csv")


def test_iter_filtered_chunks_on_in_memory_csv():
    chunks = list(iter_filtered_chunks(io.BytesIO(CSV), chunksize=2))
    assert len(chunks) == 3  # 6 rows / chunksize 2, one chunk per read
    out = pd.concat(chunks, ignore_index=True)
    assert list(out["test_id"]) == [1, 2, 5]
    assert out["test_id"].dtype == "int64"
    assert out["vehicle_id"].dtype == "int64"
    assert out["test_mileage"].dtype == "float64"
    assert out["cylinder_capacity"].dtype == "float64"
    assert out["make"].dtype == "string"
    assert out["test_mileage"].isna().sum() == 1  # blank mileage kept as NaN, not dropped


# Real rows from the February 2025 extract: DVSA escapes a quote inside a quoted
# field with a backslash (\"), not by doubling it (""). Also commas inside quotes.
CSV_QUOTES = b"""test_id,vehicle_id,test_date,test_class_id,test_type,test_result,test_mileage,postcode_area,make,model,colour,fuel_type,cylinder_capacity,first_use_date,completed_date
1059362667,500261299,2025-02-20,4,NT,F,2729,DN,PEUGEOT,"STREETZONE 50 2T 12\\"",RED,PE,49,2021-05-17,2025-02-20T12:33:56.000Z
6280,907771080,2025-02-07,4,NT,P,6417,W,JAGUAR,"HISTORIC, SPORTS",SILVER,PE,5700,1965-06-28,2025-02-07T16:04:25.000Z
7,107,2025-02-08,4,NT,P,1000,B,FORD,FIESTA,RED,DI,1200,2015-01-01,2025-02-08T10:00:00.000Z
"""


def test_backslash_escaped_quotes_do_not_shift_columns():
    out = pd.concat(iter_filtered_chunks(io.BytesIO(CSV_QUOTES)), ignore_index=True)
    assert len(out) == 3
    assert out.loc[0, "model"] == 'STREETZONE 50 2T 12"'
    assert out.loc[0, "colour"] == "RED"
    assert out.loc[0, "cylinder_capacity"] == 49
    assert out.loc[1, "model"] == "HISTORIC, SPORTS"
    assert list(out["fuel_type"]) == ["PE", "PE", "DI"]


ITEMS_CSV = b"""test_id,rfr_id,rfr_type_code,mot_test_rfr_location_type_id,dangerous_mark,completed_date
1696911325,31322,A,7,,2024-01-01T10:59:12.000Z
868674723,30844,F,3,D,2024-01-01T13:29:29.000Z
868674723,31940,A,7,,2024-01-01T13:38:44.000Z
"""


def test_iter_item_chunks_filters_by_test_id():
    from motfail.sources import item_member_name, item_zip_url, iter_item_chunks

    out = pd.concat(iter_item_chunks(io.BytesIO(ITEMS_CSV), test_ids={868674723}))
    assert list(out.columns) == ["test_id", "rfr_id", "rfr_type_code", "dangerous_mark"]
    assert list(out["rfr_id"]) == [30844, 31940]
    assert list(out["rfr_type_code"]) == ["F", "A"]
    assert out["dangerous_mark"].notna().sum() == 1
    assert item_zip_url(2024).endswith("dft_test_item_extracts_2024.zip")
    assert item_member_name(2024, 3).endswith("dft_test_item_extract_202403.csv")


def _stored_zip(files: dict[str, bytes]) -> bytes:
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_member_range_and_part_reader_return_the_exact_member_bytes():
    import zipfile

    from motfail.sources import RangeReader, member_data_range

    payload = b"".join(b"%d,row %d\n" % (i, i) for i in range(500))
    blob = _stored_zip({"a/first.csv": b"x" * 123, "a/second.csv": payload})
    calls = []

    def fetch(start: int, n: int) -> bytes:
        calls.append((start, n))
        return blob[start : start + n]

    start, length = member_data_range(zipfile.ZipFile(io.BytesIO(blob)), "a/second.csv", fetch)
    assert length == len(payload)
    reader = io.BufferedReader(RangeReader(fetch, start, length, part_size=1000))
    assert reader.read() == payload
    assert all(n <= 1000 for _, n in calls[1:])  # fetched in parts, never more than part_size
    assert len(calls) > 3


def test_member_data_range_rejects_compressed_members():
    import zipfile

    import pytest

    from motfail.sources import member_data_range

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("c.csv", b"abc" * 100)
    blob = buf.getvalue()
    with pytest.raises(ValueError, match="stored"):
        member_data_range(zipfile.ZipFile(io.BytesIO(blob)), "c.csv", lambda s, n: blob[s : s + n])
