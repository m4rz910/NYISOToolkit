import pandas as pd

from nyisotoolkit.nyisodata.nyisodata import NYISOData


class NYISOStat:
    """A class which summarizes and transforms data from the NYISOData module

    Attributes
    ----------
    year: str, int
        Year of dataset(s) to use

    Methods
    -------
    table_hourly_dataset
        Convert datasets from power to energy and resamples at hourly frequency
    table_average_day_load
        Gets total state load's average hourly value across all days of year
    table_average_day_dataset
        Gets a dataset's average hourly value across all days of year
    table_annual_energy
        Creates a table summarizing annual energy fron generation, imports, and load.
    table_instate_flow
        Gets the flow between the upstate and downstate region consistent with NYISO's definition
    table_average_instate_flow
        Gets the average hourly instate flow accross the whole year
    table_max_day_instate_flow
        Gets the particular day with the max hourly instate flow

    """

    def __init__(self, year):
        """
        Arguments
        ---------
        year: str, int
            Year to get statistics for
        """

        self.year = str(year)

    # general
    @staticmethod
    def table_hourly_dataset(dataset, year):
        """Convert datasets from power to energy and resamples at hourly frequency

        Returns
        -------
        df: Dataframe
            Energy table [GWh]
        """

        df = NYISOData(dataset=dataset, year=year).df.tz_convert("US/Eastern")  # MW
        if df.f == "5min":
            df = (df * 1 / 12).resample("h").sum() / 1000  # MW->MWh->GWh
        elif df.f == "h":
            df = df.sum() / 1000  # MW=MWh->GWh
        return df

    def table_average_day_load(self):
        """Gets total state load's average hourly value across all days of year

        Returns
        -------
        df: Dataframe
            Energy table [GWh]
        """

        df = self.table_hourly_dataset(dataset="load_5m", year=self.year)["NYCA"]
        df = pd.DataFrame(df).rename(columns={"NYCA": "Load"})
        df2 = self.table_hourly_dataset(dataset="fuel_mix_5m", year=self.year)[
            ["Wind", "Other Renewables"]
        ].sum(axis="columns")
        df["Net Load"] = df.subtract(df2, axis="index")
        return df

    def table_average_day_dataset(self):
        """Gets a dataset's average hourly value across all days of year

        Returns
        -------
        df: Dataframe
            Energy table [GWh]
        """

        if self.dataset == "load_5m":
            df = self.table_average_day_load()
        else:
            df = self.table_hourly_dataset(dataset=self.dataset, year=self.year)
        df = df.groupby(df.index.hour).mean()
        return df

    def table_annual_energy(self):
        """Creates a table summarizing annual energy fron generation, imports, and load.

        Returns
        -------
        df: Dataframe
            Energy table [TWh]
        """
        # Power [MW]
        load = NYISOData(dataset="load_5m", year=self.year).df.tz_convert("US/Eastern")[
            "NYCA"
        ]
        fuel_mix = NYISOData(dataset="fuel_mix_5m", year=self.year).df.tz_convert(
            "US/Eastern"
        )
        imports = NYISOData(dataset="interface_flows_5m", year=self.year).df.tz_convert(
            "US/Eastern"
        )
        imports.drop(
            ("External Flows", "SCH - HQ IMPORT EXPORT", "Flow (MW)"),
            axis="columns",
            inplace=True,
        )  # HQ Net is a subset of another external flow
        imports = imports.loc[:, ("External Flows", slice(None), "Flow (MW)")]

        # Energy Converstion [MWh] and Resampling By Summing Energy
        load = (load * 1 / 12).sum(axis="index").sum() / (10 ** 6)  # MW->MWh->TWh
        fuel_mix = (fuel_mix * 1 / 12).sum(axis="index") / (10 ** 6)
        imports = (imports * 1 / 12).sum(axis="index").sum() / (10 ** 6)

        fuel_mix = fuel_mix.to_frame()
        fuel_mix = fuel_mix.rename(columns={0:f'Historic ({self.year})'}).sort_values(f'Historic ({self.year})', ascending=False)
        
        #reorder carbon free resources first
        carbon_free_resources = ['Hydro','Wind','Other Renewables','Nuclear']
        df = fuel_mix.loc[carbon_free_resources]
        df = pd.concat(
            [
                df,
                fuel_mix.loc[
                    [ind for ind in fuel_mix.index if ind not in carbon_free_resources]
                ],
            ]
        )

        df.loc["Total Generation"] = fuel_mix.sum()
        df.loc["Total Renewable Generation"] = fuel_mix.loc[
            ["Hydro", "Other Renewables", "Wind"]
        ].sum()
        df.loc["Total Carbon-Free Generation"] = fuel_mix.loc[
            ["Nuclear", "Hydro", "Other Renewables", "Wind"]
        ].sum()
        df.loc["Imports"] = imports
        df.loc["Total Generation + Imports"] = (
            df.loc["Imports"] + df.loc["Total Generation"]
        )
        df.loc["Load"] = load
        df[f"Historic ({self.year}) [% of Load]"] = df / load * 100
        return df

    # Interface Flows
    def table_instate_flow(self):
        """Gets the flow between the upstate and downstate region consistent with NYISO's definition
        Returns
        -------
        df: Dataframe
            Energy table [GWh]
        """
        df = NYISOData(dataset="interface_flows_5m", year=self.year).df.tz_convert(
            "US/Eastern"
        )
        df = df[("Internal Flows", "TOTAL EAST", "Flow (MW)")]
        df = (df * 1 / 12).resample("h").sum() / 1000  # MW->MWh->GWh
        return df

    def table_average_day_instate_flow(self):
        """Gets the average hourly instate flow accross the whole year

        Returns
        -------
        df: Dataframe
            Energy table [GWh]
        """
        df = self.table_instate_flow()  # GWh
        df = df.groupby(df.index.hour).mean()
        return df

    def table_max_day_instate_flow(self):
        """Gets the particular day with the max hourly instate flow

        Returns
        -------
        df: Dataframe
            Energy table [GWh]
        """
        df = self.table_instate_flow()  # GWh
        date = df.idxmax()
        df = df.loc[date.strftime("%Y-%m-%d")]
        df.index = df.index.hour
        return df    