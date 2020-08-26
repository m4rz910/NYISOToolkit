# -*- coding: utf-8 -*-

from nyisodata import NYISOData, EXTERNAL_TFLOWS_MAP
import pandas as pd


class NYISOStat:
    #general
    @staticmethod
    def table_hourly_dataset(dataset,year):
        df = NYISOData(dataset=dataset,year=year).df.tz_convert('US/Eastern') #MW
        df = (df * 1/12).resample('H').sum()/1000  #MW->MWh->GWh
        return df
    
    @staticmethod
    def table_average_day_dataset(dataset, year):
        if dataset == 'load_5m':
            df = NYISOStat.table_average_day_load(year)
        else:
            df = NYISOStat.table_hourly_dataset(dataset=dataset,year=year)
        df = df.groupby(df.index.hour).mean()
        return df
        
    @staticmethod
    def table_average_day_load(year):
        df = NYISOStat.table_hourly_dataset(dataset='load_5m',year=year)['NYCA']
        df = pd.DataFrame(df).rename(columns={'NYCA':'Load'})
        df2 = NYISOStat.table_hourly_dataset(dataset='fuel_mix_5m',year=year)[['Wind','Other Renewables']].sum(axis='columns')
        df['Net Load'] = df.subtract(df2,axis='index')
        return df
        
    @staticmethod
    def table_annual_energy(year='2019'):
        """
        Units: TWh
        """
        #Power [MW]
        load = NYISOData(dataset='load_5m',year=year).df.tz_convert('US/Eastern')['NYCA']
        fuel_mix = NYISOData(dataset='fuel_mix_5m',year=year).df.tz_convert('US/Eastern')
        imports = NYISOData(dataset='interface_flows_5m', year='2019').df.tz_convert('US/Eastern')
        imports = imports[imports['Interface Name'].isin(EXTERNAL_TFLOWS_MAP.values())]['Flow (MW)']
    
        #Energy Converstion [MWh] and Resampling By Summing Energy
        load = (load * 1/12).sum(axis='index').sum()/(10**6) #MW->MWh->TWh
        fuel_mix = (fuel_mix * 1/12).sum(axis='index')/(10**6) 
        imports = (imports * 1/12 ).sum(axis='index').sum()/(10**6)
        
        fuel_mix = fuel_mix.to_frame()
        fuel_mix = fuel_mix.rename(columns={0:f'Historic ({year})'}).sort_values(f'Historic ({year})', ascending=False)
        
        #reorder carbon free resources first
        carbon_free_resources = ['Nuclear','Hydro','Other Renewables','Wind']
        df = fuel_mix.loc[carbon_free_resources]
        df = pd.concat([df, fuel_mix.loc[[ind for ind in fuel_mix.index if ind not in carbon_free_resources]]])
        
        df.loc['Total Generation'] = fuel_mix.sum()
        df.loc['Total Renewable Generation'] = fuel_mix.loc[['Hydro','Other Renewables','Wind']].sum()
        df.loc['Total Carbon-Free Generation'] = fuel_mix.loc[['Nuclear','Hydro','Other Renewables','Wind']].sum()
        df.loc['Net Imports'] = imports
        df.loc['Total Generation + Net Imports'] = df.loc['Net Imports'] + df.loc['Total Generation']
        df.loc['Load'] = load
        df[f'Historic ({year}) [% of Load]'] = df/load*100
        return df
    
    #Interface Flows
    @staticmethod
    def table_instate_flow(year):
        """Flow between Upstate and Downstate"""
        df = NYISOData('interface_flows_5m',year).df.tz_convert('US/Eastern')
        df = df[df['Interface Name']=='TOTAL EAST']['Flow (MW)']
        df = (df * 1/12).resample('H').sum()/1000  #MW->MWh->GWh
        return df
    
    @staticmethod
    def table_average_day_instate_flow(year):
        df = NYISOStat.table_instate_flow(year) #GWh
        df = df.groupby(df.index.hour).mean() 
        return df
    
    @staticmethod
    def table_max_day_instate_flow(year):
        df = NYISOStat.table_instate_flow(year) #GWh
        date = df.idxmax()
        df = df.loc[date.strftime("%Y-%m-%d")]
        df.index = df.index.hour
        return df 