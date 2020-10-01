# -*- coding: utf-8 -*-

from nyisodata import NYISOData
import pandas as pd

e_tflows_reg_map = {'HQ CHATEAUGUAY': 'Upstate',
                    'HQ CEDARS': 'Upstate',
                    'NPX NEW ENGLAND (NE)': 'Downstate',
                    'NPX 1385 NORTHPORT (NNC)': 'Downstate',
                    'NPX CROSS SOUND CABLE (CSC)': 'Downstate', #long island
                    'IESO': 'Upstate',
                    'PJM KEYSTONE':'Upstate', #assumption
                    'PJM HUDSON TP':'Downstate', 
                    'PJM NEPTUNE':'Downstate', #long island
                    'PJM LINDEN VFT':'Downstate'}

if __name__ == '__main__':

    loadh  = NYISOData(dataset='load_h', year='2019').df.sum(axis='index')['NYCA']
    load  = (NYISOData(dataset='load_5m', year='2019').df*(1/12)).sum(axis='index')['NYCA']
    fuel  = (NYISOData(dataset='fuel_mix_5m', year='2019').df*(1/12)).sum(axis='index').sum()
    flows = NYISOData(dataset='interface_flows_5m', year='2019').df
    
    flows = flows[['Flow (MW)','Interface Name']] 
    flows = flows[flows['Interface Name'].isin(e_tflows_reg_map.keys())][['Interface Name','Flow (MW)']]
    flows = (flows['Flow (MW)']*(1/12)).sum()
    
    df = pd.Series({'loadh':loadh,
                    'load':load,
                    'fuel':fuel,
                    'flows':flows,
                    'load-flows':load-flows})
