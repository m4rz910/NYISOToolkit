import pathlib as pl

import pandas as pd
import pytz
import requests
import zipfile
from datetime import datetime
import io

from . import utils

# from . import DataQuality

STORAGE_DIR = pl.Path(pl.Path(__file__).resolve().parent, 'storage')
DATABASE_DIR = pl.Path(STORAGE_DIR, 'databases')

class NYISOData:
    """A class used to download and construct a local database from the NYISO.

    Attributes
    ----------
    df: Dataframe
        Dataframe containing NYISO data post-processed ready for use
    dataset: str
        Name of a supported dataset found in 'nyisodata/dataset_url_map.yml'
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
    DATABASE_DIR: Pathlib Object
        Path to directory within the storage_dir that will store the finalized databases
    dataset_details: Namedtuple
        Namedtuple containing dataset details from 'dataset_url_map.yml'

    Methods
    -------
    config 
        Creates the download_dir and DATABASE_DIR directories if they don't exist
    main
        Handles logic for downloading data and constructing or reading finalized database
    get_raw_data
        Downloads and unzips raw CSV's from NYISO Website month by month
    """

    def __init__(self, dataset, year,
                 redownload=False, reconstruct=False,
                 create_csv=False):
        """Creates a local database based on dataset name and year stored in UTC.

        Parameters
        ----------
        dataset: str
            Name of a supported dataset found in 'dataset_url_map.yml'
        year: str
            Dataset year in Eastern Standard Time
            
        TODO: update: bool, optional
            A flag to stop the automatic downloading of new data from the current year (default is True)
        redownload: bool, optional
            A flag used to redownload CSV files (default is False)
        reconstruct: bool, optional
            A flag used to reconstruct database from downloaded CSV files (default is False)
        create_csv: bool, optional
            A flag used to save the database in a CSV file (default is False)
            Pickle databases are the primary database because they save frequency and timezone
        """

        # Attributes
        self.df = None
        self.dataset = dataset
        self.year = str(year)
        self.redownload = redownload
        self.reconstruct = reconstruct
        self.create_csv = create_csv
        self.arg_validation()
         
        self.curr_date = datetime.now(tz=pytz.timezone('US/Eastern'))
        self.download_dir = pl.Path(STORAGE_DIR, 'raw_datafiles', self.dataset, self.year)
        self.dataset_details = utils.fetch_dataset_url_map(self.dataset)

        self.config()
        self.main()
    
    def arg_validation(self):
        """Checks if dataset and year is available"""
        if self.dataset not in SUPPORTED_DATASETS:
            raise Exception(f"{self.dataset} is not the name of a supported dataset! Supported datasets include:{SUPPORTED_DATASETS}")
        elif (int(self.year)<=2017) or (int(self.year)>utils.current_year()):
            raise Exception(f"{self.year} is not supported. Only 2018-{utils.current_year()} is supported.")

    def config(self):
        """Creates the download_dir and DATABASE_DIR directories if they don't exist"""
        for dir_ in [self.download_dir, DATABASE_DIR]:
            dir_.mkdir(parents=True, exist_ok=True)

    def main(self):
        """Handles logic for downloading data and constructing or reading finalized database"""
        file_ = pl.Path(DATABASE_DIR, f'{self.year}_{self.dataset}.pkl')
        if not file_.exists() or self.redownload or self.reconstruct:
            if not file_.exists() or self.redownload:
                self.get_raw_data()
            # TODO: DataQuality(dataset=self.dataset, year=self.year).fix_issues()
            self.construct_database()
        else:
            self.df = pd.read_pickle(file_)

    def get_raw_data(self):
        """Downloads and unzips raw CSV's from NYISO Website month by month"""
        month_range = utils.fetch_months_to_download(self.curr_date, self.year)
        print(f"Downloading {self.year} {self.dataset}...", end="")
        for month in month_range:  # Download and extract all csv files month by month
            r = requests.get(self.dataset_details.url.format(month))
            if r.ok:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(self.download_dir)
            else:
                print(f"Warning: Request failed for {month} with status: {r.status_code}")  # TODO: log this
        print("Completed!")

    def construct_database(self):
        """Constructs database from raw datafiles and saves it in UTC"""
        # Determine expected timestamps for dataset
        self.curr_date = datetime.now(tz=pytz.timezone("US/Eastern"))  # update current time after download
        start, end = utils.fetch_ts_start_end(
             self.curr_date, self.year, self.dataset_details.f
        )
        timestamps = pd.date_range(
            start, end, freq=self.dataset_details.f, tz="US/Eastern"
        )

        # Construct Database
        files = sorted(pl.Path(self.download_dir).glob("*.csv"))
        if not files:
            print("Warning: No raw datafiles found!")
            return  # skip the rest
        else:
            if self.dataset in ['lbmp_dam_h_refbus','lbmp_rt_h_refbus']:
                 frames = [(pd.read_csv(file,header=None)
                           .rename(columns={0:'Time Stamp', 1:'Name', 2:'ID?', 3:'Marginal Cost of Energy', 4:'Loss?', 5:'Cong?'})
                           .set_index('Time Stamp')
                           .drop(columns='Name') # theres only the reference bus and we cant
                           )
                          for file in files]  # Concatenate all CSVs into a DataFrame
            else:
                frames = [pd.read_csv(file, index_col=0) for file in files]  # Concatenate all CSVs into a DataFrame
            df = pd.concat(frames, sort=False)
            df.index = pd.to_datetime(df.index)

            if ("Time Zone" in df.columns) or (self.dataset_details.col is None):
                # handle index timezone inconsistencies
                if "Time Zone" in df.columns:  # Make index timezone aware (US/Eastern)
                    df = df.tz_localize("US/Eastern", ambiguous=df["Time Zone"] == "EST")
                elif self.dataset_details.col is None:  # there is no need to pivot
                    df = df.tz_localize("US/Eastern", ambiguous="infer")
                df = df.sort_index(axis="index").tz_convert("UTC")  # Convert to UTC so that pivot can work without throwing error for duplicate indices
                if "Time Zone" in df.columns:  # make stacked columns
                    df.drop(columns=['Time Zone','PTID'],errors='ignore', inplace=True)
                    if self.dataset_details.val_col is None:
                            df = df.pivot(
                            columns=self.dataset_details.col
                        )
                    else:
                        df = df.pivot(
                            columns=self.dataset_details.col,
                            values=self.dataset_details.val_col,
                        )
                df = df.resample(self.dataset_details.f).mean()
                df = utils.check_and_interpolate_nans(df)
            else:  # When there is no timezone column and there is 'stacked' data ()
                frames = []
                for ctype, subdf in df.groupby(by=self.dataset_details.col):
                    subdf = subdf.tz_localize(
                        "US/Eastern", ambiguous="infer"
                    ).tz_convert("UTC")
                    subdf = subdf.resample(self.dataset_details.f).mean(numeric_only=True)
                    subdf = utils.check_and_interpolate_nans(subdf)
                    subdf.loc[:, self.dataset_details.col] = ctype
                    frames.append(subdf)
                df = pd.concat(frames)
                
                if isinstance(self.dataset_details.val_col,list):
                    df = df.pivot(
                        columns=self.dataset_details.col,
                        values=self.dataset_details.val_col,
                    )

            df = self.dataset_adjustments(df) # Dataset specific adjustments
            df.sort_index(inplace=True) # sort index such that slicing works
            df = df.tz_convert("US/Eastern").loc[start:end]  # Convert back to US/Eastern to select time
 
            # Checks
            assert timestamps[~timestamps.isin(df.index)].empty, f"Index is missing data! {timestamps[~timestamps.isin(df.index)]}"
            assert (~df.isnull().values.any()),"NaNs Found! Resampling and interpolation should have handled this."

            # Save and return dataset in UTC
            df = df.tz_convert('UTC')
            filepath = pl.Path(DATABASE_DIR, f'{self.year}_{self.dataset}.pkl')
            df.to_pickle(filepath)  # pickle will contains timezone and frequency information
            if self.create_csv:
                df.to_csv(filepath)
            self.df = df

    def dataset_adjustments(self, df):
        if self.dataset_details.type == "load":
            df["NYCA"] = df.sum(axis="columns")  # Calculate statewide load based on interpolated values
        if self.dataset == "load_forecast_h":
            df.rename(columns={"NYISO":"NYCA"}, inplace=True)
        if self.dataset_details.type == "interface_flows":
            # remap external interface names to match website
            df["Interface Name"] = (df["Interface Name"]
                                    .map(EXTERNAL_TFLOWS_MAP)
                                    .fillna(df["Interface Name"])
            )
            df = df.rename(
                columns={
                    "Flow (MWH)": "Flow (MW)",
                    "Positive Limit (MWH)": "Positive Limit (MW)",
                    "Negative Limit (MWH)": "Negative Limit (MW)",
                }
            )
            df = df.pivot(columns="Interface Name")  # pivot into better form
            df = df.swaplevel(axis="columns")  # add external/internal flows level
            f = (
                lambda x: "External Flows"
                if x in EXTERNAL_TFLOWS_MAP.values()
                else "Internal Flows"
            )
            df.columns = pd.MultiIndex.from_tuples(
                [(f(c[0]),) + c for c in df.columns]
            )
        return df

def construct_databases(years, datasets, redownload=False, reconstruct=False, create_csv=False):
    """Constructs all databases for selected years"""
    for dataset in datasets:
        for year in years:
            NYISOData(
                dataset=dataset,
                year=year,
                redownload=redownload,
                reconstruct=reconstruct,
                create_csv=create_csv,
            )

def table_load_weighted_price(year, rt):
    """Calculates the state-wide average energy price by weighting the zonal prices by their load."""
    if rt:
        load_df = (NYISOData(dataset="load_5m", year=year).df*1/12).tz_convert('US/Eastern') # MW->MWh
        lbmp_df = NYISOData(dataset="lbmp_rt_5m", year=year).df.tz_convert('US/Eastern')["LBMP ($/MWHr)"] #$/MWH
        lbmp_df = lbmp_df.loc[load_df.index,:] #for current year load reporting lags lbmp
    else:
         # Note: Load forecast is used
        load_df = NYISOData(dataset="load_forecast_h", year=year).df.tz_convert('US/Eastern') # MW=MWh 
        lbmp_df = NYISOData(dataset="lbmp_dam_h", year=year).df.tz_convert('US/Eastern')["LBMP ($/MWHr)"] #$/MWH
        load_df = load_df.loc[lbmp_df.index,:] # load forecast leads lbmp, so shorten it
    
    assert load_df.index.isin(lbmp_df.index).any(), "Indices are not the same"
    assert load_df.index.shape[0]==lbmp_df.index.shape[0], (
            "Indices are not the same size {} {} {}".format(load_df[~load_df.index.isin(lbmp_df.index)].index,
                                                            load_df.index.shape[0],
                                                            lbmp_df.index.shape[0])
                                                            )
    return pd.DataFrame((load_df*lbmp_df).sum(axis="columns")/load_df["NYCA"]) #$/MWh

EXTERNAL_TFLOWS_MAP = {
    "SCH - HQ - NY": "HQ CHATEAUGUAY",
    "SCH - HQ_CEDARS": "HQ CEDARS",
    "SCH - HQ_IMPORT_EXPORT": "SCH - HQ IMPORT EXPORT",  # subset of HQ Chateauguay
    "SCH - NE - NY": "NPX NEW ENGLAND (NE)",
    "SCH - NPX_1385": "NPX 1385 NORTHPORT (NNC)",
    "SCH - NPX_CSC": "NPX CROSS SOUND CABLE (CSC)",
    "SCH - OH - NY": "IESO",
    "SCH - PJ - NY": "PJM KEYSTONE",
    "SCH - PJM_HTP": "PJM HUDSON TP",
    "SCH - PJM_NEPTUNE": "PJM NEPTUNE",
    "SCH - PJM_VFT": "PJM LINDEN VFT",
}

SUPPORTED_DATASETS = [
    "load_h",
    "load_5m",
    "load_forecast_h",
    "interface_flows_5m",
    "fuel_mix_5m",
    "lbmp_dam_h",
    "lbmp_rt_5m",
    'lbmp_dam_h_refbus',
    'lbmp_rt_h_refbus',
    "asp_dam",
    "asp_rt",
]
