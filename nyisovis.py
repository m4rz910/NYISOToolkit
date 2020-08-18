# -*- coding: utf-8 -*-
from nyisodata import NYISOData, EXTERNAL_TFLOWS_MAP
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import yaml
import pathlib as pl

plt.style.use('seaborn-whitegrid')
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams["axes.grid"] = True
plt.rcParams['axes.edgecolor'] = '.15'

# Legend Colors
c_dir = pl.Path(__file__).resolve().parent
leg_path = pl.Path(c_dir, 'legend_colors.yaml')
with open(leg_path) as file: legend_deets = yaml.load(file, Loader=yaml.FullLoader)

class NYISOVis:

    @staticmethod
    def clcpa_carbon_free(year='2019', sort=False, f='D', out_dir = pl.Path(c_dir,'visualizations')):
        #Power [MW]
        load = NYISOData(dataset='load_5m',year=year).df.tz_convert('US/Eastern')['NYCA']
        fuel_mix = NYISOData(dataset='fuel_mix_5m',year=year).df.tz_convert('US/Eastern')
        imports = NYISOData(dataset='interface_flows_5m', year='2019').df.tz_convert('US/Eastern')
        imports = imports[imports['Interface Name'].isin(EXTERNAL_TFLOWS_MAP.values())]['Flow (MW)']
    
        #Energy Converstion [MWh] and Resampling By Summing Energy
        load = (load * 1/12).resample(f).sum()  
        fuel_mix = (fuel_mix * 1/12).resample(f).sum()   
        imports = (imports * 1/12 ).resample(f).sum()
        
        #Add Imports to fuel mix
        fuel_mix['Net Imports'] = imports
    
        #Calculating Energy Fraction [%]
        ef = fuel_mix.div(load, axis='index') * 100
        carbonfree_sources = ['Hydro','Wind','Other Renewables','Nuclear']
        ef['percent_carbon_free'] = ef[carbonfree_sources].sum(axis='columns')
                    
        #Plot Generation
        fig, ax = plt.subplots(figsize=(10,5))
        if sort:
            df = ef.sort_values('percent_carbon_free', ascending=True)[carbonfree_sources]
            df.reset_index(inplace=True, drop=True)
        else:
            df = ef[carbonfree_sources] #plot only carbon free resources
        df.plot.area(ax=ax,
                     color=[legend_deets.get(x, '#333333') for x in df.columns],
                     alpha=0.9, grid=True, lw=0)
        plt.ylabel('% of Load Served by NY CO$_2$e-Free Generation')
        
        #Plot Import Line
        gen_imp = ef[carbonfree_sources+['Net Imports']].sum(axis='columns')
        gen_imp.plot.line(ax=ax,linestyle='dotted', linewidth=1, color='k', label='Total + Net Imports')
        
        #Plot Average
        avg=ef[carbonfree_sources].sum(axis='columns').mean()
        plt.axhline(y=avg, color='k', linestyle='dashed')
        vert_disp= 0.05
        horizontal_p = 0.65
        plt.text(horizontal_p, avg/100-vert_disp,
                 'Average: {:.1f}%'.format(avg), transform=ax.transAxes,
                 bbox=dict(boxstyle="round",ec='black', fc='lightgray'))
        if not sort:
            #Plot CLCPA RE Goal
            plt.axhline(y=70, color='k', linestyle='solid')
            plt.text(horizontal_p, 70/100-vert_disp,
                     '2030 Renewable Energy Goal', transform=ax.transAxes,
                     bbox=dict(boxstyle="round",ec='black', fc='lightgray'))
            #Plot CLCPA RE Goal
            plt.axhline(y=100, color='k', linestyle='solid')
            plt.text(horizontal_p, 1-vert_disp,
                     '2040 Zero Emission Goal', transform=ax.transAxes,
                     bbox=dict(boxstyle="round",ec='black', fc='lightgray'))
    
        if sort:
            plt.xlabel('Days of the Year')
            ax.legend(ncol=4, bbox_to_anchor=(0.9, -0.05), fancybox=True, shadow=False)
            #ax.get_xaxis().set_visible(False)
            plt.xticks([])
            plt.xlim(ef.index[0], ef.index[-1])
            #daily average
            avg=ef[carbonfree_sources].sum(axis='columns').mean()
            plt.axhline(y=avg, color='k', linestyle='dashed')
            plt.text(0.1, 0.75,'Average: {:.1f}%'.format(avg), transform=ax.transAxes)
            #save
            file = pl.Path(out_dir, f'{year}_clcpa_carbon_free_sorted.png')
            plt.savefig(file, dpi=300, bbox_inches='tight')
            
        else:
            #Legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels),ncol=5, bbox_to_anchor=(1, -0.05), fancybox=True, shadow=False)
            #Axes
            plt.ylim(0,101)
            plt.xlim(ef.index[0], ef.index[-1])
            plt.xlabel('')
            plt.ylabel('% of Load Served by NY CO$_2$e-Free Generation')
            if f!='M':
                locator = mdates.MonthLocator()
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
                plt.xticks(rotation=0)
                for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
                    tick.tick1line.set_markersize(3)
                    tick.tick2line.set_markersize(3)
                for tick in ax.xaxis.get_major_ticks(): tick.label1.set_horizontalalignment('center')
            
            #save
            file = pl.Path(out_dir, f'{year}_clcpa_carbon_free.png')
            plt.savefig(file, dpi=300, bbox_inches='tight')
            
if __name__ == '__main__':
    NYISOVis.annual_summary(year='2019', f='D')
    #NYISOVis.clcpa_carbon_free(year='2019', sort=False, f='D')