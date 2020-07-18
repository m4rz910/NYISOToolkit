# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:37:16 2020

@author: m4rz910
"""

import pathlib as pl
import pandas as pd
import requests, zipfile, io
from datetime import datetime, timedelta
import pytz

class NYISOData:
    """
    Note: Pickle Database created includes data from LOCAL TIME, but the actual timeseries therein is saved in UTC. 
    
    Arguments:
        - dataset: name of dataset need (Currently supported: load_h, load_5m, fuel_mix_5m,'interface_flows_5m')
        - year: Year of data needed (in local time)
        - reconstruct: Whether to redownload NYISO data and reconstruct pickle database
    """
    
    def __init__(self, dataset, year, storage_dir=pl.Path(__file__).resolve().parent, reconstruct=False):
        print('Working on {} for {}...'.format(dataset,year))
        
        self.df = None #dataframe containing dataset of choice
        self.dataset = dataset #name of dataset
        self.year = str(year) #force to be string if not already
        self.curr_date = datetime.now(tz=pytz.timezone('US/Eastern')) #datetime object for current time
        
        self.storage_dir = storage_dir
        self.download_dir = None # directory that raw csv will be extracted to
        self.output_dir = None # directory where files will be saved
        
        self.type = None # dataset type (load, fuel_mix, etc)
        self.f = None    # dataset frequency
        self.col = None # column name in csv that contains regions
        self.val_col = None # column name in csv that contains values

        self.main(reconstruct)
        
    def main(self, reconstruct):
        #Set up download and output folders
        self.download_dir = pl.Path(self.storage_dir, 'raw_nysio_datafiles', self.dataset, self.year)
        self.output_dir = pl.Path(self.storage_dir, 'nyiso_databases')
        for directory in [self.download_dir, self.output_dir]:
            pl.Path(directory).mkdir(parents=True, exist_ok=True)
        
        #Check whether to get new data and construct new DB
        if (not pl.Path(self.output_dir,'{}_{}.pkl'.format(self.year,self.dataset)).is_file()) or reconstruct:
            self.get_raw_data()
            self.construct_database_utc()
        else:
            print('{}_{}.pkl exists'.format(self.year,self.dataset))
        print('Done\n')
            
    def get_raw_data(self):
        """
        Timezone: NYISO Data is in local time: US/Eastern
        
        "Integrated Real-Time Actual Load is posted after each hour and represents the timeweighted hourly load for each zone"
        - Frequency: Hourly (Sometimes they may miss or do higher)
        - Datetime Convention: Start of hour
        
        "Real-Time Actual Load posts the actual measured load for each RTD interval (5 minutes) by zone. 
        Actual loads are calculated as generation plus net interchange for each zone, based on real-time telemetered data."
        - Frequency: 5 mins (Sometimes they may miss or do higher)
        - Datetime Convention: End of 5 mins
        
        Energy Mix
        - Datetime convention: End of 5 mins
        
        
        Interface flows
        - Positive and Negative limits are currently not being pulled
        - Datetime Convention: End of 5 mins
        """
        #Config
        base_url = 'http://mis.nyiso.com/public/csv/'
        dataset_url_map = {'load_5m':{
                               'type':'load', #dataset type
                               'url':'{}pal/{}pal_csv.zip'.format(base_url,{}),
                               'f'  :'5T',
                               'col':'Name', #column containing regions
                               'val_col':'Load'}, #column containing values in raw file
                           'load_h':{
                               'type':'load',
                               'url':'{}palIntegrated/{}palIntegrated_csv.zip'.format(base_url,{}),
                               'f':'H',
                               'col':'Name', #column containing regions
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
        self.type = dataset_url_map[self.dataset]['type']
        self.f = dataset_url_map[self.dataset]['f']
        self.col = dataset_url_map[self.dataset]['col']
        self.val_col = dataset_url_map[self.dataset]['val_col']
        
        #Download Data
        print('Downloading Data from NYISO...')
        dataset_url = dataset_url_map[self.dataset]['url']
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
            
        for month in month_range: #Download data month by month
            r = requests.get(dataset_url.format(month))
            if r.ok:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(self.download_dir)
            else:
                print('Warning: Request failed for {}'.format(month))
                
    
    def construct_database_utc(self):
        print('Constructing DB...')
        frames = []
        files = sorted(pl.Path(self.download_dir).glob('*.csv'))
        if files:
            for file in files:
                frames.append(pd.read_csv(file,index_col=0))
            df = pd.concat(frames, sort=False)
            try:
                df.index = pd.to_datetime(df.index).tz_localize('US/Eastern', ambiguous=df['Time Zone']=='EST') # data is in local time
            except KeyError:
                df.index = pd.to_datetime(df.index).tz_localize('US/Eastern', ambiguous='infer')

            #Convert to UTC so that pivot can work without 'duplicate' indices (due to DST)
            df = df.sort_index(axis='index').tz_convert('UTC') 
            df = check_and_interpolate_nans(df)
            df = df.pivot(columns=self.col, values=self.val_col) # make columns
            print('Resampling...')
            df = df.resample(self.f).mean()     
            df = check_and_interpolate_nans(df)
            if self.type == 'load':
                df['NYCA'] = df.sum(axis='columns') #NY Control Area: done last, based on interpolated values

            #Convert back to Eastern to select time period based on local time
            df = df.tz_convert('US/Eastern') 
            if self.curr_date.year == int(self.year):
                if self.f == '5T':
                    start = '{}-01-01 00:05:00'.format(self.year) #Datetime Convention: End
                    end   =  (self.curr_date + timedelta(hours=-1)).strftime('%Y-%m-%d %H:00:00') #todo: get latest minute info
                elif self.f == 'H':
                    start = '{}-01-01 00:00:00'.format(self.year) #Datetime Convention: Start
                    end   = (self.curr_date + timedelta(hours=-1)).strftime('%Y-%m-%d %H:00:00')
            elif self.curr_date.year > int(self.year):
                if self.f == '5T':
                    start = '{}-01-01 00:05:00'.format(self.year) #Datetime Convention: End
                    end   = '{}-01-01 00:00:00'.format(int(self.year)+1)
                elif self.f == 'H':
                    start = '{}-01-01 00:00:00'.format(self.year) #Datetime Convention: Start
                    end   = '{}-12-31 23:00:00'.format(self.year)
            df = df.loc[start:end] 
            
            #Check all the dates that should exist
            #todo: handle if youre dealing with the current year, which has not finished yet
            date_range = pd.date_range(start=start, end=end, freq=self.f, tz='US/Eastern') 
            assert date_range[~date_range.isin(df.index)].empty, 'Warning: Index is missing data! {}'.format(date_range[~date_range.isin(df.index)])
            assert ~df.isnull().values.any(), 'Warning: Nans Found! There shouldnt be any because of resampling and interpolation.'
            
            #Save in UTC
            df = df.tz_convert('UTC')
            df.to_pickle(pl.Path(self.output_dir,'{}_{}.pkl'.format(self.year, self.dataset)))
            self.df = df
        else:
            print('Warning: No raw datafiles found!')
            
def check_and_interpolate_nans(df):
    """If there are nan in the data, then interpolate"""
    if df.isnull().values.any(): 
        print('Warning: {} Nans found... interpolating'.format(df.isna().sum().sum()))
        df.interpolate(method='linear', inplace=True)
    else:
        print('Yay! No interpolation needed!')
    return df

def construct_all_databases(year,reconstruct):
    for dataset in ['fuel_mix_5m','load_h', 'load_5m','interface_flows_5m']:
        NYISOData(dataset=dataset, year=year, reconstruct=reconstruct)

    
if __name__ == '__main__':
    #Construct all db
    #for year in ['2013','2019']:
    #    construct_all_databases(year,reconstruct=True)
    
    year='2019'
    dataset = 'interface_flows_5m'
    NYISOData(dataset=dataset, year=year)
    
    current_dir = pl.Path(__file__).resolve().parent
    df = pd.read_pickle(pl.Path(current_dir,'nyiso_databases','{}_{}.pkl'.format(year, dataset))).tz_convert('US/Eastern')