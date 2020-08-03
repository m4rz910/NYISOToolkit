import datetime
import pytest

import utils

def test_fetch_months_to_download_general():
    cur_date = datetime.datetime(2020, 8, 2)
    input_year = 2019

    expected = [
        '20191201', '20200101', '20200201', '20200301', '20200401',
        '20200501', '20200601', '20200701', '20200801'
    ]

    actual = utils.fetch_months_to_download(cur_date=cur_date, input_year=input_year)

    # sort for testing ease
    expected = sorted(expected)
    actual = sorted(actual)

    assert actual == expected

