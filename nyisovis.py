# -*- coding: utf-8 -*-
from nyisodata import NYISOData, EXTERNAL_TFLOWS_MAP
from nyisostat import NYISOStat
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as opy
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import yaml
import pathlib as pl
import os
import numpy as np
import pandas as pd

# Figure Configuration
plt.style.use(['seaborn-whitegrid'])
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams["axes.grid"] = True
plt.rcParams['axes.edgecolor'] = '.15'

c_dir = pl.Path(__file__).resolve().parent
figure_config_path = pl.Path(c_dir, 'legend_colors.yaml')
with open(figure_config_path) as file:
    all_configs = yaml.load(file, Loader=yaml.FullLoader)
    LEGEND_DEETS = all_configs['legend_colors']

class NYISOVis:
        
    @staticmethod
    def fig_energy(year='2019', f='D', out_dir = pl.Path(c_dir,'visualizations')):
        if f in ['Y']:
            raise ValueError('Frequency Not Supported!')
        
        #Power [MW]
        load = NYISOData(dataset='load_5m',year=year).df.tz_convert('US/Eastern')['NYCA']
        fuel_mix = NYISOData(dataset='fuel_mix_5m',year=year).df.tz_convert('US/Eastern')
        imports = NYISOData(dataset='interface_flows_5m', year=year).df.tz_convert('US/Eastern')
        imports = imports[imports['Interface Name'].isin(EXTERNAL_TFLOWS_MAP.values())]['Flow (MW)']
        
        dfs = {'load':load, 'fuel_mix':fuel_mix, 'imports': imports} # group datasets into dictionary to apply 
    
        def power_to_energy(df):
            """Energy Converstion [MWh] and Resampling By Summing Energy"""
            return (df * 1/12).resample(f).sum()/1000  #MW->MWh->GWh
        dfs = {k: power_to_energy(v) for k, v in dfs.items()}
        
        dfs['fuel_mix']['Net Imports'] = dfs['imports'] #add imports to fuel mix
        del dfs['imports'] #remove imports from the dfs list
        
        order = ['Nuclear','Hydro','Other Renewables','Wind','Natural Gas','Dual Fuel',
                 'Other Fossil Fuels', 'Net Imports'] 
        dfs['fuel_mix'] = dfs['fuel_mix'][order] #fix order
        
        def month_adj_object(df):
            """Adjust index for months and make index objects to label correctly"""
            if f == 'M':
                df.index = df.index.shift(-1,'M').shift(1,'D')
            df.index = df.index.astype('O')
            return df       
        dfs = {k: month_adj_object(v) for k, v in dfs.items()}
        
        #Plots
        fig, ax = plt.subplots(figsize=(10,5), dpi=300)
        dfs['fuel_mix'].plot.area(ax=ax,
                                  color=[LEGEND_DEETS.get(x, '#333333') for x in dfs['fuel_mix'].columns],
                                  alpha=0.9, lw=0) #fuel mix
        dfs['load'].plot.line(ax=ax, linestyle='dashed', linewidth=1, color='k', label='Load') #load line
        
        #Legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fancybox=True, shadow=False)
        #Axes
        ax.set(title=f'Historic ({year})',
               xlabel=None, ylabel='Energy [GWh]',
               xlim=[dfs['fuel_mix'].index[0], dfs['fuel_mix'].index[-1]],
               ylim=None)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
        plt.setp(ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines(), markersize=3)
        
        #Save
        file = pl.Path(out_dir, f'{year}_energy_{f}.png')
        plt.savefig(file, bbox_inches='tight', transparent=True)
        
                
    @staticmethod
    def fig_clcpa_carbon_free(year='2019', f='D',
                              out_dir = pl.Path(c_dir,'visualizations')):
        """
        Figure Inspiration: NYISO Power Trends 2020 - Figure 12: Production of In-State Renewables & Zero-Emission Resources Relative to 2019 Load 
        """
        if f in ['Y']:
            raise ValueError('Frequency Not Supported!')
        
        #Power [MW]
        load = NYISOData(dataset='load_5m',year=year).df.tz_convert('US/Eastern')['NYCA']
        fuel_mix = NYISOData(dataset='fuel_mix_5m',year=year).df.tz_convert('US/Eastern')
        imports = NYISOData(dataset='interface_flows_5m', year=year).df.tz_convert('US/Eastern')
        imports = imports[imports['Interface Name'].isin(EXTERNAL_TFLOWS_MAP.values())]['Flow (MW)']
    
        #Energy Converstion [MWh] and Resampling By Summing Energy
        load = (load * 1/12).resample(f).sum()  
        fuel_mix = (fuel_mix * 1/12).resample(f).sum()   
        imports = (imports * 1/12 ).resample(f).sum()
        fuel_mix['Net Imports'] = imports #Add Imports to fuel mix
    
        #Calculating Carbon-free Fraction [%]
        ef = fuel_mix.div(load, axis='index') * 100
        carbonfree_sources = ['Hydro','Other Renewables','Wind','Nuclear']
        ef['percent_carbon_free'] = ef[carbonfree_sources].sum(axis='columns')
                    
        #Plot Carbon-free Fraction
        df = ef[carbonfree_sources] #plot only carbon free resources
        if f == 'M':
            df.index = df.index.shift(-1,'M').shift(1,'D')
            ef.index = ef.index.shift(-1,'M').shift(1,'D')
        df.index = df.index.astype('O')
        fig, ax = plt.subplots(figsize=(10,5), dpi=300)
        df.plot.area(ax=ax,
                     color=[LEGEND_DEETS.get(x, '#333333') for x in df.columns],
                     alpha=0.9, lw=0)
       
        #Plot Import Line
        gen_imp = ef[carbonfree_sources+['Net Imports']].sum(axis='columns')
        gen_imp.index = gen_imp.index.astype('O')
        gen_imp.plot.line(ax=ax,linestyle='dotted',
                          linewidth=1, color='k', label='Total + Net Imports')
    
        #Plot Goals and Progress
        data = [[t for t in carbonfree_sources if t!='Nuclear'],
                carbonfree_sources]
        averages = ['Renewable: {:.0f}% + Net Imports: {:.0f}% (70% by 2030)',
                    'Carbon-Free: {:.0f}% + Net Imports: {:.0f}% (100% by 2040)']
        colors = ['limegreen','lawngreen']
        h_distances = [0.05, 0.05]
        for t,l,c,h in zip(data, averages,colors,h_distances):
            avg = fuel_mix[t].sum(axis='index').sum() / load.sum(axis='index') * 100
            avg_imp = fuel_mix[t+['Net Imports']].sum(axis='index').sum() / load.sum(axis='index') * 100
            ax.axhline(y=avg, xmax=h, color='k', linestyle='solid', lw=1)
            ax.text(h, avg/100, l.format(avg, avg_imp), 
                     bbox=dict(boxstyle='round',ec='black',fc=c, alpha=0.9),
                     transform=ax.transAxes)
        #Legend
        #handles, labels = ax.get_legend_handles_labels()
        ax.legend(loc='upper center',bbox_to_anchor=(0.45, -0.05),
                  ncol=5, fancybox=True, shadow=False)
        
        #Axes
        ax.set(title=f'Historic ({year})',
               xlabel=None, ylabel='Percent of Load Served by Carbon-Free Energy',
               xlim=[df.index[0], df.index[-1]], ylim=[0, 100])
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
        plt.setp(ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines(), markersize=3)
        
        #Save
        file = pl.Path(out_dir, f'{year}_clcpa_carbon_free_{f}.png')
        plt.savefig(file, bbox_inches='tight', transparent=True)
            
    @staticmethod
    def fig_carbon_free_year(year='2019', out_dir = pl.Path(c_dir,'visualizations')):
        stats = NYISOStat.table_annual_energy(year=year)
        df = stats[f'Historic ({year}) [% of Load]'].drop(index=['Total Renewable Generation',
                                                                 'Total Carbon-Free Generation',
                                                                 'Total Generation',
                                                                 'Total Generation + Net Imports',
                                                                 'Load'])
        df = df.to_frame().T
        #Plot
        fig, ax = plt.subplots(figsize=(4,8), dpi=300)
        df.plot.bar(stacked=True, ax=ax,
                    color=[LEGEND_DEETS.get(x, '#333333') for x in df.columns],
                    alpha=0.9)
        #Averages and Goals
        # Carbon Free Line
        perc = stats[f'Historic ({year}) [% of Load]'].loc['Total Carbon-Free Generation']
        ax.axhline(y=perc, color='k', linestyle='dashed',
                    label='CO$_2$e-Free Generation')
        ax.text(-0.0755, perc/100,'{:.0f}'.format(perc), transform=ax.transAxes)
        # Carbon Free + Imports Line
        perc = stats[f'Historic ({year}) [% of Load]'].loc[['Total Carbon-Free Generation',
                                                            'Net Imports']].sum()
        ax.axhline(y=perc, color='k', linestyle='dotted',
                    label='CO$_2$e-Free Generation + Net Imports')
        ax.text(-0.0755, perc/100,'{:.0f}'.format(perc), transform=ax.transAxes)
        
        #Legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels),
                  loc='right', bbox_to_anchor=(2.2, 0.5),
                  ncol=1, fancybox=True, shadow=False)
        #Axes
        ax.set(title=f'Historic ({year})',
               xlabel=year, ylabel='Percent of Load [%]',
               xlim=None, ylim=None)
        plt.xticks([])
        plt.savefig(pl.Path(out_dir,f'{year}_carbon_free_year.png'))

    
    def fig_carbon_free_years(years):
        """Todo: stacked area chart over time using nyisostat annual summary"""
        return
    
    def net_load():
        """Todo: Load vs Net Load Shape"""
        return
            
if __name__ == '__main__':
    for year in ['2019']:
        NYISOVis.fig_carbon_free_year(year=year)
        for f in ['D','M']:
            NYISOVis.fig_energy(year=year, f=f)
            NYISOVis.fig_clcpa_carbon_free(year=year, f=f)
