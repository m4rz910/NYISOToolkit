import pytest

from nyisotoolkit import construct_databases, SUPPORTED_DATASETS


def test_datasets(years=["2019"]):
    datasets = SUPPORTED_DATASETS
    construct_databases(
        years=years,
        datasets=datasets,
        redownload=False,
        reconstruct=True,
        create_csv=False,
    )
