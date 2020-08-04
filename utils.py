from collections import namedtuple

import pandas as pd
from plumbum import local

BASE_URL = 'http://mis.nyiso.com/public/csv/'
YAML_FILE = 'dataset_url_map.yml'

dataset_details = namedtuple('dataset_details', ['dataset', 'type', 'url', 'f', 'col', 'val_col'])


def fetch_months_to_download(cur_date, year_to_collect):
    """
    Determine correct months to download. NYISO datasets are labled based on month start

    :cur_date: current date -> datetime;
    :year_to_collect: year to validate against -> int;
    """
    year_to_collect = int(year_to_collect)  # fail fast
    output_fmt = '%Y%m%d'

    if year_to_collect > cur_date.year:
        assert False, 'Error: Year to collect is greater than current year'

    range_end = f'{cur_date.year}-{cur_date.month}-01' if cur_date.year == year_to_collect \
        else f'{year_to_collect + 1}-01-01'

    return pd.date_range(
        start=f'{year_to_collect - 1}-12-01',  # start at last month of previous year
        end=range_end,
        freq='MS'
    ).strftime(output_fmt)


def check_and_interpolate_nans(df):
    """
    If there are NANs in the data, interpolate

    :df: pandas dataframe to process -> pd.DataFrame
    """
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f'Note: {nan_count} Nans found... interpolating')
        df.interpolate(method='linear', inplace=True)
    return df


def fetch_dataset_url_map(dataset):
    """
    Reads yaml for dataset and returns namedtuple containing details

    :dataset: name of dataset -> str;
    """
    path = local.path(__file__).dirname / YAML_FILE
    yml = open_yml(path)[dataset]

    return dataset_details(
        dataset,
        yml['type'],
        BASE_URL + yml['url']['pre'] + "/{}" + yml['url']['post'],
        yml['f'],
        yml.get('col'),
        yml.get('val_col')
    )




def open_yml(filepath_):
    import yaml
    with open(filepath_, 'r') as stream:
        return yaml.safe_load(stream)