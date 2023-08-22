from nyisotoolkit import construct_databases, SUPPORTED_DATASETS, NYISOData


def test_datasets(years=[i for i in range(2018,2024)]):
    datasets = SUPPORTED_DATASETS
    construct_databases(
        years=years,
        datasets=datasets,
        redownload=False,
        reconstruct=True,
        create_csv=False,
    )

if __name__ == "__main__":
    test_datasets()