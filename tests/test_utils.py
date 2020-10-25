import datetime
import pytest
import pandas as pd
import pathlib as pl

from nyisotoolkit.nyisodata import utils

def test_fetch_months_to_download_general():
    cur_date = datetime.datetime(2020, 8, 2)
    year_to_collect = 2019

    expected = [
        '20181201', '20190101', '20190201', '20190301', '20190401', '20190501',
        '20190601', '20190701', '20190801', '20190901', '20191001', '20191101',
        '20191201', '20200101'
    ]

    actual = utils.fetch_months_to_download(cur_date=cur_date, year_to_collect=year_to_collect)

    # sort for testing ease
    expected = sorted(expected)
    actual = sorted(actual)

    assert actual == expected


def test_fetch_months_to_download_bad_input():
    cur_date = datetime.datetime(2020, 8, 2)

    with pytest.raises(ValueError):
        utils.fetch_months_to_download(cur_date=cur_date, year_to_collect=2021)


def test_test_fetch_months_to_download_same_year():
    cur_date = datetime.datetime(2020, 8, 2)

    expected = [
        '20191201', '20200101', '20200201', '20200301', '20200401', '20200501',
        '20200601', '20200701', '20200801'
    ]

    actual = utils.fetch_months_to_download(cur_date=cur_date, year_to_collect=2020)

    # sort for testing ease
    expected = sorted(expected)
    actual = sorted(actual)

    assert actual == expected


def test_fetch_dataset_url_map_general():
    dataset = "lbmp_rt_5m"
    expected = utils.dataset_details(
        dataset,
        'lbmp',
        utils.BASE_URL + "realtime" + "/{}" + "realtime_zone_csv.zip",
        "5T",
        "Name",
        None
    )
    assert utils.fetch_dataset_url_map(dataset) == expected


def test_fetch_dataset_url_map_failure():
    with pytest.raises(KeyError):
        utils.fetch_dataset_url_map('bad_key')


def test_build_db_ts_range_general():
    cur_date = datetime.datetime(2020, 8, 5)
    request_year = 2019
    frequency = 'H'

    exp_start = f'{request_year}-01-01 00:00:00'
    exp_end = f'{request_year}-12-31 23:00:00'

    actual = utils.fetch_ts_start_end(cur_date, request_year, frequency)
    expected = (exp_start, exp_end)

    # sort for testing ease
    expected = sorted(expected)
    actual = sorted(actual)

    assert actual == expected


def test_build_db_ts_range_bad_year():
    cur_date = datetime.datetime(2020, 8, 5)
    request_year = 2021  # later than current
    frequency = 'H'

    with pytest.raises(ValueError):
        utils.fetch_ts_start_end(cur_date=cur_date, request_year=request_year, frequency=frequency)


def test_build_db_ts_range_unsupported_frequency():
    cur_date = datetime.datetime(2020, 8, 5)
    request_year = 2019
    frequency = 'bad_frequency_string'

    with pytest.raises(NotImplementedError):
        utils.fetch_ts_start_end(cur_date, request_year, frequency)
