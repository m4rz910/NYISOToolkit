# -*- coding: utf-8 -*-
from nyisodata import NYISOData, EXTERNAL_TFLOWS_MAP
from nyisostat import NYISOStat
import matplotlib.dates as mdates
import matplotlib.pyplot as plt 
import yaml
import pathlib as pl
import os

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
            print('Frequency Not Supported!')
            return None
        
        #Power [MW]
        load = NYISOData(dataset='load_5m',year=year).df.tz_convert('US/Eastern')['NYCA']
        fuel_mix = NYISOData(dataset='fuel_mix_5m',year=year).df.tz_convert('US/Eastern')
        imports = NYISOData(dataset='interface_flows_5m', year=year).df.tz_convert('US/Eastern')
        imports = imports[imports['Interface Name'].isin(EXTERNAL_TFLOWS_MAP.values())]['Flow (MW)']
    
        #Energy Converstion [MWh] and Resampling By Summing Energy
        load = (load * 1/12).resample(f).sum()/1000  #MWh->GWh
        fuel_mix = (fuel_mix * 1/12).resample(f).sum()/1000   
        imports = (imports * 1/12 ).resample(f).sum()/1000 
        
        #Add Imports to fuel mix
        fuel_mix['Net Imports'] = imports
            
        #Plot Generation
        order = ['Nuclear','Hydro','Other Renewables','Wind','Natural Gas','Dual Fuel','Other Fossil Fuels', 'Net Imports']
        df = fuel_mix[order] 
        fig, ax = plt.subplots(figsize=(10,5), dpi=300)
        plt.title(f'Historic ({year})')
        df.plot.area(ax=ax,
                     color=[LEGEND_DEETS.get(x, '#333333') for x in df.columns],
                     alpha=0.9, lw=0)
        plt.ylabel('% of Load Served by NY CO$_2$e-Free Generation')
        
        #Load
        load.plot.line(ax=ax,linestyle='dashed', linewidth=1, color='k', label='Load')
        
        #Legend
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3),
                  ncol=4, fancybox=True, shadow=False)
        #Axis
        plt.xlabel(''); plt.ylabel('Energy [GWh]')
        plt.xlim(df.index[0], df.index[-1])
        plt.ylim(0,900)
        if f!='M':
            locator = mdates.MonthLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            plt.xticks(rotation=0)
            for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
                tick.tick1line.set_markersize(3)
                tick.tick2line.set_markersize(3)
            for tick in ax.xaxis.get_major_ticks(): tick.label1.set_horizontalalignment('center')
        #Save
        file = pl.Path(out_dir, f'{year}_energy.png')
        plt.savefig(file, bbox_inches='tight')
        
                
    @staticmethod
    def fig_clcpa_carbon_free(year='2019', sort=False, f='D', out_dir = pl.Path(c_dir,'visualizations')):
        """
        Inspiration: NYISO Power Trends 2020 - Figure 12: Production of In-State Renewables & Zero-Emission Resources Relative to 2019 Load 
        """
        if f in ['Y']:
            print('Frequency Not Supported!')
            return None
        
        #Power [MW]
        load = NYISOData(dataset='load_5m',year=year).df.tz_convert('US/Eastern')['NYCA']
        fuel_mix = NYISOData(dataset='fuel_mix_5m',year=year).df.tz_convert('US/Eastern')
        imports = NYISOData(dataset='interface_flows_5m', year=year).df.tz_convert('US/Eastern')
        imports = imports[imports['Interface Name'].isin(EXTERNAL_TFLOWS_MAP.values())]['Flow (MW)']
    
        #Energy Converstion [MWh] and Resampling By Summing Energy
        load = (load * 1/12).resample(f).sum()  
        fuel_mix = (fuel_mix * 1/12).resample(f).sum()   
        imports = (imports * 1/12 ).resample(f).sum()
        
        #Add Imports to fuel mix
        fuel_mix['Net Imports'] = imports
    
        #Calculating Energy Fraction [%]
        ef = fuel_mix.div(load, axis='index') * 100
        carbonfree_sources = ['Nuclear','Hydro','Other Renewables','Wind']
        ef['percent_carbon_free'] = ef[carbonfree_sources].sum(axis='columns')
                    
        #Plot Generation
        fig, ax = plt.subplots(figsize=(10,5), dpi=300)
        plt.title(f'Historic ({year})')
        if sort:
            df = ef.sort_values('percent_carbon_free', ascending=True)[carbonfree_sources]
            df.reset_index(inplace=True, drop=True)
        else:
            df = ef[carbonfree_sources] #plot only carbon free resources
        df.plot.area(ax=ax,
                     color=[LEGEND_DEETS.get(x, '#333333') for x in df.columns],
                     alpha=0.9, lw=0)
        plt.ylabel('% of Load Served by NY CO$_2$e-Free Generation')
        
        #Plot Import Line
        gen_imp = ef[carbonfree_sources+['Net Imports']].sum(axis='columns')
        gen_imp.plot.line(ax=ax,linestyle='dotted', linewidth=1, color='k', label='Total + Net Imports')
        
        #Plot Averages and Goals
        data = [[t for t in carbonfree_sources if t!='Nuclear'],
                [t for t in carbonfree_sources if t!='Nuclear']+['Net Imports'],
                carbonfree_sources, 
                carbonfree_sources+['Net Imports']]
        averages = ['Renewable: {:.0f}% (70x2030)',
                    'Renewable + Net Imports: {:.0f}% (70x2030)',
                    'CO$_2$e-Free: {:.0f}% (100x2040)',
                    'CO$_2$e-Free + Net Imports: {:.0f}% (100x2040)']
        colors = ['limegreen','limegreen','lawngreen','lawngreen']
        h_distances = [0.05,0.05,0.55,0.55]
        for t,l,c,h in zip(data, averages,colors,h_distances):
            avg = ef[t].sum(axis='columns').mean()
            plt.axhline(y=avg, xmax=h, color='k', linestyle='solid', lw=1.5)
            vert_disp= 0.05
            plt.text(h, avg/100-vert_disp, l.format(avg), 
                     bbox=dict(boxstyle="round",ec='black',fc=c, alpha=0.9),
                     transform=ax.transAxes)

        if sort:
            #Legend
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15),
                      ncol=4, fancybox=True, shadow=False)
            #Axes
            plt.xlabel('Days of the Year')
            plt.xticks([])
            plt.xlim(ef.index[0], ef.index[-1])
            #daily average
            avg=ef[carbonfree_sources].sum(axis='columns').mean()
            plt.axhline(y=avg, color='k', linestyle='dashed')
            plt.text(0.1, 0.75,'Average: {:.0f}%'.format(avg), transform=ax.transAxes)
            #save
            file = pl.Path(out_dir, f'{year}_clcpa_carbon_free_sorted.png')
            plt.savefig(file)
        else:
            #Legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(loc='lower center',bbox_to_anchor=(0.45, -0.15),
                      ncol=5, fancybox=True, shadow=False)
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
            #Save
            file = pl.Path(out_dir, f'{year}_clcpa_carbon_free.png')
            plt.savefig(file, bbox_inches='tight')
            
    @staticmethod
    def fig_carbon_free_year(year='2019', out_dir = pl.Path(c_dir,'visualizations')):
        stats = NYISOStat.table_annual_energy(year=year)
        df = stats[f'Historic ({year}) [% of Load]'].drop(index=['Total Carbon-Free Generation',
                                                         'Total Generation',
                                                         'Net Imports',
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
        plt.axhline(y=perc, color='k', linestyle='dashed',
                    label='CO$_2$e-Free Generation')
        plt.text(-0.0755, perc/100,'{:.0f}'.format(perc), transform=ax.transAxes)
        # Carbon Free + Imports Line
        perc = stats[f'Historic ({year}) [% of Load]'].loc[['Total Carbon-Free Generation',
                                                    'Net Imports']].sum()
        plt.axhline(y=perc, color='k', linestyle='dotted',
                    label='CO$_2$e-Free Generation + Net Imports')
        plt.text(-0.0755, perc/100,'{:.0f}'.format(perc), transform=ax.transAxes)
        
        #Legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels),
                  loc='right', bbox_to_anchor=(2.2, 0.5),
                  ncol=1, fancybox=True, shadow=False)
        #Axes
        plt.xlabel(year); plt.ylabel('Percent of Load [%]')
        plt.xticks([]); plt.ylim(0,100)
        plt.savefig(pl.Path(out_dir,f'{year}_carbon_free_year.png'))
        return df
            
    def fig_carbon_free_years(years):
        """Todo: stacked area chart over time using nyisostat annual summary"""
        return
    
    def net_load():
        """Todo: Load vs Net Load Shape"""
        return
            
if __name__ == '__main__':
    NYISOVis.fig_energy(year='2019',f='D')
    NYISOVis.fig_carbon_free_year(year='2019')
    NYISOVis.fig_clcpa_carbon_free(year='2019', sort=False, f='D')