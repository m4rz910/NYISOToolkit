# -*- coding: utf-8 -*-

import pathlib as pl
import pandas as pd
import requests, zipfile, io
from datetime import datetime, timedelta
import pytz

supported_datasets = ['load_h', 'load_5m',
                      'fuel_mix_5m',
                      'interface_flows_5m']

class NYISOData:
    
    def __init__(self, dataset, year,
                 reconstruct=False, storage_dir=pl.Path('__file__').resolve().parent):
        """
        Creates a local pickle database based on dataset name and year stored in UTC.
        
        Arguments:
            - dataset: name of dataset need (Currently supported: load_h, load_5m, fuel_mix_5m,'interface_flows_5m')
            - year: Year of data needed (in local time)
            - reconstruct: If true, redownload NYISO data and reconstruct pickle database
            - storage_dir: The directory that the raw csvs and pickle databases will be stored
        """
        
        print('Working on {} for {}...'.format(dataset,year))
        self.df = None #dataframe containing dataset of choice
        self.dataset = dataset #name of dataset
        self.year = str(year) #force to be string if not already
        self.curr_date = datetime.now(tz=pytz.timezone('US/Eastern')) #datetime object for current time
        
        self.storage_dir = storage_dir #where databases and raw csvs will be stored
        self.download_dir = None # directory that raw csv will be extracted to
        self.output_dir = None # directory where files will be saved
        
        self.dataset_url_map = None #contains information
        self.type = None # dataset type (load, fuel_mix, etc)
        self.f = None    # dataset frequency
        self.col = None # column name in csv that contains regions
        self.val_col = None # column name in csv that contains values

        self.config() #sets some of the class attributes
        self.main(reconstruct)
        
    def config(self):
        """Sets important class attributes and creates directories for storing files"""
        
        base_url = 'http://mis.nyiso.com/public/csv/'
        self.dataset_url_map = {'load_5m':{
                                   'type':'load', #dataset type
                                   'url':'{}pal/{}pal_csv.zip'.format(base_url,{}),
                                   'f'  :'5T',
                                   'col':'Name', #csv column containing regions
                                   'val_col':'Load'}, # csv column containing values
                               'load_h':{
                                   'type':'load',
                                   'url':'{}palIntegrated/{}palIntegrated_csv.zip'.format(base_url,{}),
                                   'f':'H',
                                   'col':'Name', #csv column containing regions
                                   'val_col':'Integrated Load'},
                               'fuel_mix_5m':{
                                   'type':'fuel_mix',
                                   'url':'{}rtfuelmix/{}rtfuelmix_csv.zip'.format(base_url,{}),
                                   'f':'5T',
                                   'col':'Fuel Category',
                                   'val_col':'Gen MW'},
                               'interface_flows_5m':{
                                   'type':'interface_flows',
                                   'url':'{}ExternalLimitsFlows/{}ExternalLimitsFlows_csv.zip'.format(base_url,{}),
                                   'f':'5T',
                                   'col':'Interface Name',
                                   'val_col':'Flow (MWH)'}}
        self.type = self.dataset_url_map[self.dataset]['type']
        self.f = self.dataset_url_map[self.dataset]['f']
        self.col = self.dataset_url_map[self.dataset]['col']
        self.val_col = self.dataset_url_map[self.dataset]['val_col']
        
        #Set up download and output folders
        self.download_dir = pl.Path(self.storage_dir, 'raw_datafiles', self.dataset, self.year)
        self.output_dir = pl.Path(self.storage_dir, 'databases')
        for directory in [self.download_dir, self.output_dir]:
            pl.Path(directory).mkdir(parents=True, exist_ok=True)
        
    def main(self, reconstruct): 
        """Decides whether to download new data and (re)make database or just read existing one."""
        
        #Check whether to get new data and construct new DB
        if (not pl.Path(self.output_dir,'{}_{}.pkl'.format(self.year,self.dataset)).is_file()) or reconstruct:
            #self.get_raw_data()
            self.construct_database()
        else:
            print('{}_{}.pkl exists'.format(self.year,self.dataset))
            self.df = pd.read_pickle(pl.Path(self.output_dir,'{}_{}.pkl'.format(self.year, self.dataset)))
        print('Done\n')
            
    def get_raw_data(self):
        """Downloads raw CSV's from NYISO Website"""
        
        #Determine correct months to download
        print('Downloading Data from NYISO...')
        if self.curr_date.year == int(self.year):
            month_range = pd.date_range(start= '{}-01-01'.format(self.year),
                                        end  = '{}-{}-01'.format(self.curr_date.year, self.curr_date.month),
                                        freq = 'MS').strftime('%Y%m%d') # NYISO are labled based on month start)
        elif self.curr_date.year > int(self.year):
            month_range = pd.date_range(start= '{}-01-01'.format(self.year),
                                        end  = '{}-01-01'.format(int(self.year)+1),
                                        freq = 'MS').strftime('%Y%m%d')
        else:
            assert False, 'Error: Year greater than current year entered'
        
        #Download and extract all csv files month by month
        dataset_url = self.dataset_url_map[self.dataset]['url']
        for month in month_range: 
            r = requests.get(dataset_url.format(month))
            if r.ok:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(self.download_dir)
            else:
                print('Warning: Request failed for {}\n{}'.format(month, r.status_code))
                #todo: add reason
    
    def construct_database(self):
        """Constructs database from raw datafiles and saves it in UTC"""
        #Determine expected timestamps for dataset
        self.curr_date = datetime.now(tz=pytz.timezone('US/Eastern')) #update current time after download
        #If the requested year's data is the current year, then get partial dataset
        if self.curr_date.year == int(self.year):
            if self.f == '5T':
                start = '{}-01-01 00:05:00'.format(self.year) #Datetime Convention: End
                end   =  (self.curr_date + timedelta(hours=-1)).strftime('%Y-%m-%d %H:00:00') #todo: get latest minute info
            elif self.f == 'H':
                start = '{}-01-01 00:00:00'.format(self.year) #Datetime Convention: Start
                end   = (self.curr_date + timedelta(hours=-1)).strftime('%Y-%m-%d %H:00:00')
        #If previous year data is requested get the full year's dataset
        elif self.curr_date.year > int(self.year):
            if self.f == '5T':
                start = '{}-01-01 00:05:00'.format(self.year) 
                end   = '{}-01-01 00:00:00'.format(int(self.year)+1)
            elif self.f == 'H':
                start = '{}-01-01 00:00:00'.format(self.year) 
                end   = '{}-12-31 23:00:00'.format(self.year)
        else:
            assert False, 'A year larger than the current year was queried!'
        timestamps = pd.date_range(start=start, end=end, freq=self.f, tz='US/Eastern')
        
        # Construct Database
        print('Constructing DB...')
        files = sorted(pl.Path(self.download_dir).glob('*.csv'))
        if not files:
            assert False, 'No raw datafiles found!'
        else:
            #Concatenate all CSVs into a DataFrame
            frames = [pd.read_csv(file, index_col=0) for file in files]
            df = pd.concat(frames, sort=False)
            df.index = pd.to_datetime(df.index)
            
            # Create 'Time Zone' column if the csv files don't include one
            if 'Time Zone' not in df.columns:
                #issue section
                #sol 1 (havent tested, but way too intensive)
                window_size = df[self.col].unique()
                raw = {t:subdf.pivot(columns=self.col, values=self.val_col) for t, subdf in df.rolling(len(window_size))}
                df = pd.DataFrame.from_dict(raw)
                
                
                #end of issue section
                
            #Make index timezone aware (US/Eastern)
            df = df.tz_localize('US/Eastern', ambiguous=df['Time Zone']=='EST')

            #Convert to UTC so that pivot can work without throwing error for duplicate indices (due to DST)
            df = df.sort_index(axis='index').tz_convert('UTC') 
            df = check_and_interpolate_nans(df)
            df = df.pivot(columns=self.col, values=self.val_col) # make columns
            print('Resampling...')
            df = df.resample(self.f).mean()     
            df = check_and_interpolate_nans(df)
            if self.type == 'load':
                df['NYCA'] = df.sum(axis='columns') #Calculate statewide load based on interpolated values

            #Convert back to US/Eastern to select time period based on local time
            df = df.tz_convert('US/Eastern') 
            df = df.loc[start:end] 
            
            #Check to make sure that all the expected timestamps exist
            assert timestamps[~timestamps.isin(df.index)].empty, 'Index is missing data! {}'.format(timestamps[~timestamps.isin(df.index)])
            assert ~df.isnull().values.any(), 'NANs Found! Resampling and interpolation should have handled this.'
            #Save and return dataset in UTC
            df = df.tz_convert('UTC')
            df.to_pickle(pl.Path(self.output_dir,'{}_{}.pkl'.format(self.year, self.dataset)))
            self.df = df
            
def check_and_interpolate_nans(df):
    """If there are NANs in the data, interpolate"""
    if df.isnull().values.any(): 
        print('Warning: {} Nans found... interpolating'.format(df.isna().sum().sum()))
        df.interpolate(method='linear', inplace=True)
    else:
        print('Yay! No interpolation was needed.')
    return df

def construct_all_databases(year=str(datetime.now(tz=pytz.timezone('US/Eastern')).year),
                            reconstruct=True):
    """Constructs all Databases for current year"""
    for dataset in supported_datasets:
        NYISOData(dataset=dataset, year=year, reconstruct=reconstruct)

    
if __name__ == '__main__':
    #construct_all_databases()
    
    #Construct all db
    #for year in ['2013','2019']:
    #    construct_all_databases(year,reconstruct=True)

    df = NYISOData(dataset='load_h', year='2019').df