import pandas as pd


def fetch_months_to_download(cur_date, input_year):
    """
    Determine correct months to download. NYISO datasets are labled based on month start

    :cur_date: current date -> datetime;
    :input_year: year to validate against -> int;
    """
    input_year = int(input_year)  # fail fast
    output_fmt = '%Y%m%d'

    if input_year > cur_date.year:
        assert False, 'Error: Year greater than current year entered'

    end_ = f'{cur_date.year}-{cur_date.month}-01' if cur_date.year == input_year else f'{input_year + 1}-01-01'

    return pd.date_range(
        start=f'{input_year - 1}-12-01',  # get month before
        end=end_,
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
