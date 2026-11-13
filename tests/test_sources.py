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
