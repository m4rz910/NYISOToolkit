# -*- coding: utf-8 -*-

from nyisodata import NYISOData, EXTERNAL_TFLOWS_MAP

class NYISOStat:
    
    @staticmethod
    def annual_energy_summary(year='2019'):
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
        
        df = fuel_mix.to_frame()
        df = df.rename(columns={0:'Historical'}).sort_values('Historical', ascending=False)
        
        df.loc['Total Generation'] = df.sum()
        df.loc['Net Imports'] = imports
        df.loc['Total Generation + Net Imports'] = df.loc['Net Imports'] + df.loc['Total Generation']
        df.loc['Load'] = load
        df['Historical [% of Load]'] = df/load*100
        return df