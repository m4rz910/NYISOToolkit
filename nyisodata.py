# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

import io
import pandas as pd
import pathlib as pl
import pytz
import requests
import zipfile

import utils


class NYISOData:

    def __init__(self, dataset, year,
                 reconstruct=False, create_csvs=False, storage_dir=pl.Path(__file__).resolve().parent):
        """
        Creates a local database based on dataset name and year stored in UTC.
        
        Arguments:
            - dataset: name of dataset need 
            - year: Year of data needed (in Eastern Time)
            - reconstruct: If true, redownload NYISO data and reconstruct database
            - create_csvs: whether to also save the databases as csvs (pickle dbs are used because they maintain frequency and timezone information)
            - storage_dir: The directory that the raw csvs and databases will be stored
        """
        print(f'Working on {dataset} for {year}')
        self.df = None #dataframe containing dataset of choice
        self.dataset = dataset #name of dataset
        self.year = str(year) #force to be string if not already
        self.curr_date = datetime.now(tz=pytz.timezone('US/Eastern')) #datetime object for current time
        self.create_csvs=create_csvs #if false will oncreate pickle dbs
        
        self.storage_dir = storage_dir #where databases and raw csvs will be stored
        self.download_dir = None # directory that raw csv will be extracted to
        self.output_dir = None # directory where files will be saved
        
        self.dataset_url_map = None #contains information
        self.type = None # dataset type (load, fuel_mix, etc)
        self.f = None    # dataset frequency
        self.col = None # column name in csv that contains regions
        self.val_col = None # column name in csv that contains values

        self.config()  # sets some of the class attributes
        self.main(reconstruct)

    def config(self):
        """Sets important class attributes and creates directories for storing files"""
        base_url = 'http://mis.nyiso.com/public/csv/'
        brackets = "{}"  # for ease
        self.dataset_url_map = {
            'load_5m': {
                'type': 'load',  # dataset type
                'url': f'{base_url}pal/{brackets}pal_csv.zip',
                'f': '5T',
                'col': 'Name',  # csv column containing regions
                'val_col': 'Load'  # csv column containing values
            },
            'load_h': {
                'type': 'load',
                'url': '{}palIntegrated/{}palIntegrated_csv.zip'.format(base_url, '{}'),
                'f': 'H',
                'col': 'Name',  # csv column containing regions
                'val_col': 'Integrated Load'},
            'load_forecast_h': {
                'type': 'load_forecast',
                'url': '{}isolf/{}isolf_csv.zip'.format(base_url, '{}'),
                'f': 'H',
                'col': None,  # used to indicate when no column pivot is necessary
                'val_col': None},
            'fuel_mix_5m': {
                'type': 'fuel_mix',
                'url': '{}rtfuelmix/{}rtfuelmix_csv.zip'.format(base_url, '{}'),
                'f': '5T',
                'col': 'Fuel Category',
                'val_col': 'Gen MW'},
            'interface_flows_5m': {
                'type': 'interface_flows',
                'url': '{}ExternalLimitsFlows/{}ExternalLimitsFlows_csv.zip'.format(base_url, '{}'),
                'f': '5T',
                'col': 'Interface Name',
                'val_col': 'Flow (MWH)'},
            'lbmp_dam_h': {
                'type': 'lbmp',
                'url': '{}damlbmp/{}damlbmp_zone_csv.zip'.format(base_url, '{}'),
                'f': 'H',
                'col': 'Name',
                'val_col': None},
            'lbmp_rt_5m': {
                'type': 'lbmp',
                'url': '{}realtime/{}realtime_zone_csv.zip'.format(base_url, '{}'),
                'f': '5T',
                'col': 'Name',
                'val_col': None}}  # note, the column has actual units of MW, fixed in output
        self.type = self.dataset_url_map[self.dataset]['type']
        self.f = self.dataset_url_map[self.dataset]['f']
        self.col = self.dataset_url_map[self.dataset]['col']
        self.val_col = self.dataset_url_map[self.dataset]['val_col']

        # Set up download and output folders
        self.download_dir = pl.Path(self.storage_dir, 'raw_datafiles', self.dataset, self.year)
        self.output_dir = pl.Path(self.storage_dir, 'databases')
        for directory in [self.download_dir, self.output_dir]:
            pl.Path(directory).mkdir(parents=True, exist_ok=True)

    def main(self, reconstruct):
        """Decides whether to download new data and (re)make database or just read existing one."""
        # Check whether to get new data and construct new DB
        file_ = pl.Path(self.output_dir, f'{self.year}_{self.dataset}.pkl')
        if not file_.exists() or reconstruct:
            self.get_raw_data()
            self.construct_database()
        else:
            print(f'{file_.name} exists')
            self.df = pd.read_pickle(file_)
        print('Done\n')


    def get_raw_data(self):
        """Downloads raw CSV's from NYISO Website"""
        month_range = utils.fetch_months_to_download(self.curr_date.year, self.year)

        # Download and extract all csv files month by month
        print('Downloading Data from NYISO...')
        dataset_url = self.dataset_url_map[self.dataset]['url']
        for month in month_range:
            r = requests.get(dataset_url.format(month))
            if r.ok:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(self.download_dir)
            else:
                print(f'Warning: Request failed for {month}\n{r.status_code}')
                # todo: add reason

    def construct_database(self):
        """Constructs database from raw datafiles and saves it in UTC"""
        # Determine expected timestamps for dataset
        self.curr_date = datetime.now(tz=pytz.timezone('US/Eastern'))  # update current time after download
        # If the requested year's data is the current year, then get partial dataset
        if self.curr_date.year == int(self.year):
            start = '{}-01-01 00:00:00'.format(self.year)
            if self.f == '5T':
                end = (self.curr_date + timedelta(hours=-1)).strftime(
                    '%Y-%m-%d %H:00:00')  # todo: get latest minute info
            elif self.f == 'H':
                end = (self.curr_date + timedelta(hours=-1)).strftime('%Y-%m-%d %H:00:00')
        # If previous year data is requested get the full year's dataset
        elif self.curr_date.year > int(self.year):
            start = '{}-01-01 00:00:00'.format(self.year)
            if self.f == '5T':
                end = '{}-12-31 23:55:00'.format(self.year)
            elif self.f == 'H':
                end = '{}-12-31 23:00:00'.format(self.year)
        else:
            assert False, 'A year larger than the current year was queried!'
        timestamps = pd.date_range(start=start, end=end, freq=self.f, tz='US/Eastern')

        # Construct Database
        print('Constructing DB...')
        files = sorted(pl.Path(self.download_dir).glob('*.csv'))
        if not files:
            print('Warning: No raw datafiles found!')
            return  # skip the rest
        else:
            # Concatenate all CSVs into a DataFrame
            frames = [pd.read_csv(file, index_col=0) for file in files]
            df = pd.concat(frames, sort=False)
            df.index = pd.to_datetime(df.index)

            # If self.col is None then there is no need to pivot
            if ('Time Zone' in df.columns) or (self.col == None):
                # Make index timezone aware (US/Eastern)
                if 'Time Zone' in df.columns:
                    df = df.tz_localize('US/Eastern', ambiguous=df['Time Zone'] == 'EST')
                elif (self.col == None):
                    df = df.tz_localize('US/Eastern', ambiguous='infer')
                df = df.sort_index(axis='index').tz_convert('UTC')
                # Convert to UTC so that pivot can work without throwing error for duplicate indices (due to
                if 'Time Zone' in df.columns:
                    print('Pivoting Data...')
                    df = df.pivot(columns=self.col, values=self.val_col)  # make columns
                print('Resampling...')
                df = df.resample(self.f).mean()
                df = check_and_interpolate_nans(df)
            # When there is no timezone column and there is 'stacked' data
            else:
                print('Data is stacked...')
                frames = []
                for ctype, subdf in df.groupby(by=self.col):
                    subdf = subdf.tz_localize('US/Eastern', ambiguous='infer').tz_convert('UTC')
                    subdf = subdf.resample(self.f).mean()
                    subdf = check_and_interpolate_nans(subdf)
                    subdf.loc[:, self.col] = ctype
                    frames.append(subdf)
                df = pd.concat(frames)
                # check if the number of regions/interface flow name are equal
                if not (len(set(df[self.col].value_counts().values)) <= 1):
                    print('Warning: There seems to be underlying missing data.\n{}'.format(df[self.col].value_counts()))

            if self.type == 'load':
                df['NYCA'] = df.sum(axis='columns')  # Calculate statewide load based on interpolated values
            if self.type == 'interface_flows':
                # remap external interface names to match website
                df['Interface Name'] = df['Interface Name'].map(EXTERNAL_TFLOWS_MAP).fillna(df['Interface Name'])
                df = df.rename(columns={'Flow (MWH)': 'Flow (MW)',
                                        'Postitive Limit (MWH)': 'Postitive Limit (MW)',
                                        'Negative Limit (MWH)': 'Negative Limit (MW)'})

            # Convert back to US/Eastern to select time period based on local time
            df = df.tz_convert('US/Eastern')
            df = df.loc[start:end]

            # Check to make sure that all the expected timestamps exist
            assert timestamps[~timestamps.isin(df.index)].empty, 'Index is missing data! {}'.format(
                timestamps[~timestamps.isin(df.index)])
            assert ~df.isnull().values.any(), 'NANs Found! Resampling and interpolation should have handled this.'
            # Save and return dataset in UTC
            df = df.tz_convert('UTC')
            df.to_pickle(pl.Path(self.output_dir, '{}_{}.pkl'.format(self.year,
                                                                     self.dataset)))  # pickle will contains timezone and frequency information
            if self.create_csvs:
                df.to_csv(pl.Path(self.output_dir, '{}_{}.csv'.format(self.year, self.dataset)))
            self.df = df


def check_and_interpolate_nans(df):
    """If there are NANs in the data, interpolate"""
    if df.isnull().values.any():
        print('Note: {} Nans found... interpolating'.format(df.isna().sum().sum()))
        df.interpolate(method='linear', inplace=True)
    return df


def construct_databases(years, datasets, reconstruct=False, create_csvs=False):
    """Constructs all databases for selected years"""
    for dataset in datasets:
        for year in years:
            NYISOData(dataset=dataset, year=year, reconstruct=reconstruct, create_csvs=create_csvs)


EXTERNAL_TFLOWS_MAP = {'SCH - HQ - NY': 'HQ CHATEAUGUAY',
                       'SCH - HQ_CEDARS': 'HQ CEDARS',
                       'SCH - HQ_IMPORT_EXPORT': 'HQ NET',
                       'SCH - NE - NY': 'NPX NEW ENGLAND (NE)',
                       'SCH - NPX_1385': 'NPX 1385 NORTHPORT (NNC)',
                       'SCH - NPX_CSC': 'NPX CROSS SOUND CABLE (CSC)',
                       'SCH - OH - NY': 'IESO',
                       'SCH - PJ - NY': 'PJM KEYSTONE',
                       'SCH - PJM_HTP': 'PJM HUDSON TP',
                       'SCH - PJM_NEPTUNE': 'PJM NEPTUNE',
                       'SCH - PJM_VFT': 'PJM LINDEN VFT'}

SUPPORTED_DATASETS = ['load_h', 'load_5m', 'load_forecast_h',
                      'interface_flows_5m', 'fuel_mix_5m',
                      'lbmp_dam_h', 'lbmp_rt_5m']

if __name__ == '__main__':
    years = ['2019', '2013']
    datasets = SUPPORTED_DATASETS
    construct_databases(years=years, datasets=datasets, reconstruct=True, create_csvs=False)
