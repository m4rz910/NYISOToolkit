import pandas as pd
import requests
from pathlib import Path
import pytz
import os
import time

from nyisotoolkit.nyisodata import STORAGE_DIR

class NYISOCapacity:
    def __init__(self, date=pd.Timestamp.now(), wait=False) -> None:
        """
        Arguments:
            date (pandas.Timestamp): date that will be used to pull latest capacity
                report (will refer to month and year)
        """
        self.date = pd.to_datetime(date)
        self.download_dir = Path(STORAGE_DIR, 'raw_datafiles', 'capacity_reports')
        self.download_dir.mkdir(parents=True,exist_ok=True)
        self.report_name = f"ICAP-Market-Report-{self.date.month_name()}-{self.date.year}.xlsx"
        
        self.exists = None # set in get_raw_data
        
        if wait:
            pass
        else:
            self.get_raw_data()

    def get_url(self):
         
        year_to_year_code = {
            2014: 1410927,
            2015: 1410895,
            2016: 1410901,
            2017: 1410883,
            2018: 1410889,
            2019: 4266869,
            2020: 10106066,
            2021: 18170164,
            2022: 27447313,
            2023: 35397361,
            2024: 42146126,
            2025: 48997190,
            2026: 56195933,
        }
        
        year_code = year_to_year_code.get(self.date.year, None)
        if year_code is None:
            raise ValueError("Year not currently supported. Please file an issue.")
        capacity_market_base_url = f"https://www.nyiso.com/documents/20142/{year_code}"
        report_name = f"ICAP-Market-Report-{self.date.month_name()}-{self.date.year}.xlsx"
        url = f"{capacity_market_base_url}/{report_name}"
        return url, report_name

    def get_raw_data(self, repull=False, **request_kwargs):
        """Will determine whether whether the local reports pulled are usable and only pull new ones"""
        
        url, report_name = self.get_url() # pull the latest file
        self.report_file = Path(self.download_dir, report_name)

        # check does the local report file already exist locally
        if self.report_file.exists() and (os.stat(self.report_file).st_size > 0) and (not repull):
            self.exists = True
        else: # otherwise we have to pull the new files down
            
            r = requests.get(url, **request_kwargs)
            
            if (r.ok) and (len(r.content)>0): # Check if the content is not empty   
                with open(self.report_file, 'wb') as file:
                    file.write(r.content)
                self.exists = True
            else:
                raise Exception(f"Request for {url} returned an empty response.")
                self.exists = False
                self.report_file.unlink(missing_ok=True)
            
    def prices(self):
        """Pull the most recent capacity market report's market clearing prices

        Returns:
            a DataFrame of monthly capacity prices (all three auctions) for each of the four capacity localities within NYISO
        """
        if not Path(self.download_dir, self.report_name).exists():
            self.get_raw_data(self)
        
        if self.exists:  
            df = pd.read_excel(Path(self.download_dir, self.report_name),
                            sheet_name="MCP Table", header=[0, 1]).iloc[:,:13]
            df.rename(columns={"Unnamed: 0_level_0": "", "Date": ""},
                    inplace=True)
            df.set_index("", inplace=True)
            df.index.name = 'datetime'
            return df
        else:
            raise Exception(f'Data is not available - file is probably not available anymore on the website or its corrupted: {self.url}')
    
    def summary_table(self):
        if not Path(self.download_dir, self.report_name).exists():
            self.get_raw_data(self)
            
        df = pd.read_excel(Path(self.download_dir, self.report_name), engine='openpyxl',
                           sheet_name="Summary Table", header=[0, 1]).iloc[:,:17]
        df.rename(columns={"Unnamed: 0_level_0": "", "Date": ""},
                  inplace=True)
        df.set_index("", inplace=True)
        df.index.name = 'datetime'
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        return df

    def ucap_table(self):
        if not Path(self.download_dir, self.report_name).exists():
            self.get_raw_data(self)
            
        df = pd.read_excel(Path(self.download_dir, self.report_name), engine='openpyxl',
                           sheet_name="UCAP Table", header=[0, 1]).iloc[:,:17]
        df.rename(columns={"Unnamed: 0_level_0": "", "Date": ""},
                  inplace=True)
        df.set_index("", inplace=True)
        df.index.name = 'datetime'
        
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        return df
    
if __name__ == "__main__":
    o = NYISOCapacity(date='09/01/2024')
    o