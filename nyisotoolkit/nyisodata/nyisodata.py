import pathlib as pl

import pandas as pd
import pytz
import requests
import zipfile
from datetime import datetime
import io

from . import utils
#from . import DataQuality


class NYISOData:
    """A class used to download and construct a local database from the NYISO.
    
    
    Attributes
    ----------
    df: Dataframe
        Dataframe containing NYISO data post-processed ready for use
    dataset: str
        Name of a supported dataset found in 'dataset_url_map.yml'
    year: str
        Dataset year in Eastern Standard Time
    redownload: bool, optional
        A flag used to redownload CSV files (default is False)
    reconstruct: bool, optional
        A flag used to reconstruct database from downloaded CSV files (default is False)
    create_csv: bool
        A flag used to save the database in a CSV file (default is False). 
        Pickle databases are the primary database because they save frequency and timezone.
    curr_date: Datetime Object
        Datetime object of current time
    storage_dir: Pathlib Object
        Path to directory which will contain directories for finalized databases and raw CSV files
    download_dir: Pathlib Object
        Path to directory within the storage_dir that will store the raw CSV files downloaded from the NYISO
    output_dir: Pathlib Object
        Path to directory within the storage_dir that will store the finalized databases
    dataset_details: Namedtuple
        Namedtuple containing dataset details from 'dataset_url_map.yml' 
    
    Methods
    -------
    config 
        Creates the download_dir and output_dir directories if they don't exist
    main
        Handles logic for downloading data and constructing or reading finalized database  
    get_raw_data
        Downloads and unzips raw CSV's from NYISO Website month by month
    """

    def __init__(self, dataset, year,
                 redownload=False, reconstruct=False, create_csv=False):
        """Creates a local database based on dataset name and year stored in UTC.
        
        Parameters
        ----------
        dataset: str
            Name of a supported dataset found in 'dataset_url_map.yml' 
        year: str
            Dataset year in Eastern Standard Time
        redownload: bool, optional
            A flag used to redownload CSV files (default is False)
        reconstruct: bool, optional
            A flag used to reconstruct database from downloaded CSV files (default is False)
        create_csv: bool, optional
            A flag used to save the database in a CSV file (default is False)
            Pickle databases are the primary database because they save frequency and timezone  
        """
        
        #Attributes
        self.df = None 
        self.dataset = dataset
        self.year = str(year)
        self.redownload = redownload
        self.reconstruct = reconstruct
        self.create_csv = create_csv
        
        self.curr_date = datetime.now(tz=pytz.timezone('US/Eastern'))
        self.storage_dir = pl.Path(pl.Path(__file__).resolve().parent, 'storage')
        self.download_dir = pl.Path(self.storage_dir, 'raw_datafiles', self.dataset, self.year)
        self.output_dir = pl.Path(self.storage_dir, 'databases')
        self.dataset_details = utils.fetch_dataset_url_map(self.dataset)
        
        #Methods
        self.config()
        self.main()

    def config(self):
        """Creates the download_dir and output_dir directories if they don't exist"""
        for dir_ in [self.download_dir, self.output_dir]:
            dir_.mkdir(parents=True, exist_ok=True)

    def main(self):
        """Handles logic for downloading data and constructing or reading finalized database"""
        file_ = pl.Path(self.output_dir, f'{self.year}_{self.dataset}.pkl')
        if not file_.exists() or self.redownload or self.reconstruct:
            if not file_.exists() or self.redownload:
                self.get_raw_data()
            #TODO: DataQuality(dataset=self.dataset, year=self.year).fix_issues()
            self.construct_database()
        else:
            self.df = pd.read_pickle(file_)

    def get_raw_data(self):
        """Downloads and unzips raw CSV's from NYISO Website month by month"""
        month_range = utils.fetch_months_to_download(self.curr_date, self.year)
        print(f'Downloading {self.year} {self.dataset}...', end='')
        for month in month_range: # Download and extract all csv files month by month
            r = requests.get(self.dataset_details.url.format(month))
            if r.ok:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(self.download_dir)
            else:
                print(f'Warning: Request failed for {month} with status: {r.status_code}')  #TODO: log this
        print('Completed!')

    def construct_database(self):
        """Constructs database from raw datafiles and saves it in UTC"""
        # Determine expected timestamps for dataset
        self.curr_date = datetime.now(tz=pytz.timezone('US/Eastern'))  # update current time after download
        start, end = utils.fetch_ts_start_end(self.curr_date, self.year, self.dataset_details.f)
        timestamps = pd.date_range(start, end, freq=self.dataset_details.f, tz='US/Eastern')

        # Construct Database
        files = sorted(pl.Path(self.download_dir).glob('*.csv'))
        if not files:
            print('Warning: No raw datafiles found!')
            return  # skip the rest
        else:
            frames = [pd.read_csv(file, index_col=0) for file in files] # Concatenate all CSVs into a DataFrame
            df = pd.concat(frames, sort=False)
            df.index = pd.to_datetime(df.index)

            if ('Time Zone' in df.columns) or (self.dataset_details.col is None):
                if 'Time Zone' in df.columns: # Make index timezone aware (US/Eastern)
                    df = df.tz_localize('US/Eastern', ambiguous=df['Time Zone'] == 'EST')
                elif self.dataset_details.col is None: # there is no need to pivot
                    df = df.tz_localize('US/Eastern', ambiguous='infer')
                df = df.sort_index(axis='index').tz_convert('UTC') # Convert to UTC so that pivot can work without throwing error for duplicate indices
                if 'Time Zone' in df.columns: # make stacked columns
                    df = df.pivot(columns=self.dataset_details.col, values=self.dataset_details.val_col)  
                df = df.resample(self.dataset_details.f).mean()
                df = utils.check_and_interpolate_nans(df)
            else: # When there is no timezone column and there is 'stacked' data
                frames = []
                for ctype, subdf in df.groupby(by=self.dataset_details.col):
                    subdf = subdf.tz_localize('US/Eastern', ambiguous='infer').tz_convert('UTC')
                    subdf = subdf.resample(self.dataset_details.f).mean()
                    subdf = utils.check_and_interpolate_nans(subdf)
                    subdf.loc[:, self.dataset_details.col] = ctype
                    frames.append(subdf)
                df = pd.concat(frames)
                # Check if the number of regions/interface flow name are equal
                if not (len(set(df[self.dataset_details.col].value_counts().values)) <= 1):
                    print('Warning: There seems to be underlying missing data.\n{}'.format(
                        df[self.dataset_details.col].value_counts()))
            
            # Dataset specific adjustments
            if self.dataset_details.type == 'load':
                df['NYCA'] = df.sum(axis='columns')  # Calculate statewide load based on interpolated values
            if self.dataset_details.type == 'interface_flows':
                # remap external interface names to match website
                df['Interface Name'] = df['Interface Name'].map(EXTERNAL_TFLOWS_MAP).fillna(df['Interface Name'])
                df = df.rename(columns={'Flow (MWH)': 'Flow (MW)',
                                        'Positive Limit (MWH)': 'Positive Limit (MW)',
                                        'Negative Limit (MWH)': 'Negative Limit (MW)'})
                df = df.pivot(columns='Interface Name') #pivot into better form
                df = df.swaplevel(axis='columns') #add external/internal flows level
                f = lambda x: 'External Flows' if x in EXTERNAL_TFLOWS_MAP.values() else 'Internal Flows'
                df.columns = pd.MultiIndex.from_tuples([(f(c[0]),) + c for c in df.columns])
                
            df = df.tz_convert('US/Eastern').loc[start:end] #Convert back to US/Eastern to select time

            # Checks
            assert timestamps[~timestamps.isin(df.index)].empty, 'Index is missing data! {}'.format(
                timestamps[~timestamps.isin(df.index)])
            assert ~df.isnull().values.any(), 'NaNs Found! Resampling and interpolation should have handled this.'
            
            # Save and return dataset in UTC
            df = df.tz_convert('UTC')
            filepath = pl.Path(self.output_dir, f'{self.year}_{self.dataset}.pkl')
            df.to_pickle(filepath)  # pickle will contains timezone and frequency information
            if self.create_csv:
                df.to_csv(filepath)
            self.df = df

def construct_databases(years, datasets,
                        redownload=False, reconstruct=False, create_csv=False):
    """Constructs all databases for selected years"""
    for dataset in datasets:
        for year in years:
            NYISOData(dataset=dataset, year=year,
                      redownload=redownload, reconstruct=reconstruct, create_csv=create_csv)

EXTERNAL_TFLOWS_MAP = {'SCH - HQ - NY': 'HQ CHATEAUGUAY',
                       'SCH - HQ_CEDARS': 'HQ CEDARS',
                       'SCH - HQ_IMPORT_EXPORT': 'SCH - HQ IMPORT EXPORT', #subset of HQ Chateauguay 
                       'SCH - NE - NY': 'NPX NEW ENGLAND (NE)',
                       'SCH - NPX_1385':'NPX 1385 NORTHPORT (NNC)',
                       'SCH - NPX_CSC': 'NPX CROSS SOUND CABLE (CSC)',
                       'SCH - OH - NY': 'IESO',
                       'SCH - PJ - NY': 'PJM KEYSTONE',
                       'SCH - PJM_HTP': 'PJM HUDSON TP',
                       'SCH - PJM_NEPTUNE': 'PJM NEPTUNE',
                       'SCH - PJM_VFT': 'PJM LINDEN VFT'}

SUPPORTED_DATASETS = ['load_h', 'load_5m', 'load_forecast_h',
                      'interface_flows_5m', 'fuel_mix_5m',
                      'lbmp_dam_h', 'lbmp_rt_5m',
                      'asp_dam', 'asp_rt']