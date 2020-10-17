import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import yaml
import pathlib as pl

from nyisotoolkit.nyisodata.nyisodata import NYISOData
from nyisotoolkit.nyisostat.nyisostat import NYISOStat

# Figure Configuration
plt.style.use(['seaborn-whitegrid'])
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams["axes.grid"] = True
plt.rcParams['axes.edgecolor'] = '.15'

# Legend Colors
c_dir = pl.Path(__file__).resolve().parent
figure_config_path = pl.Path(c_dir, 'legend_colors.yml')
with open(figure_config_path) as file:
    all_configs = yaml.load(file, Loader=yaml.FullLoader)
    LEGEND_DEETS = all_configs['legend_colors']
    
# List of Carbon Free Resources
CARBONFREE_SOURCES = ['Hydro','Other Renewables','Wind','Nuclear']


class NYISOVis:
        
    @staticmethod
    def dfs_energy(year='2019', f='D'):
        if f in ['Y']:
            raise ValueError('Frequency Not Supported!')
        
        #Power [MW]
        load = NYISOData(dataset='load_5m', year=year).df.tz_convert('US/Eastern')['NYCA']
        fuel_mix = NYISOData(dataset='fuel_mix_5m', year=year).df.tz_convert('US/Eastern')
        imports = NYISOData(dataset='interface_flows_5m', year=year).df.tz_convert('US/Eastern')
        imports = imports.loc[:, ('External Flows', slice(None), 'Flow (MW)')]
        imports.drop(('External Flows', 'SCH - HQ_IMPORT_EXPORT', 'Flow (MW)'), axis='columns', inplace=True) #'SCH - HQ_IMPORT_EXPORT' is a subset of another external flow
        imports = imports.sum(axis='columns')
        
        dfs = {'load':load, 'fuel_mix':fuel_mix, 'imports': imports} # group datasets into dictionary to apply 
        def power_to_energy(df):
            """Energy Converstion [MWh] and Resampling By Summing Energy"""
            return (df * 1/12).resample(f).sum()/1000  #MW->MWh->GWh
        dfs = {k: power_to_energy(v) for k, v in dfs.items()}
        
        dfs['fuel_mix']['Imports'] = dfs['imports'] #add imports to fuel mix
        del dfs['imports'] #remove imports from the dfs list
        
        order = ['Nuclear','Hydro','Other Renewables','Wind','Natural Gas','Dual Fuel',
                 'Other Fossil Fuels', 'Imports'] 
        dfs['fuel_mix'] = dfs['fuel_mix'][order] #fix order
        
        def month_adj_object(df):
            """Adjust index for months and make index objects to label correctly"""
            if f == 'M':
                df.index = df.index.shift(-1,'M').shift(1,'D')
                df = df/1000 # GWh->TWh
            df.index = df.index.astype('O')
            return df       
        dfs = {k: month_adj_object(v) for k, v in dfs.items()}
        return dfs
        
    @staticmethod
    def fig_energy(year='2019', f='D', out_dir = pl.Path(c_dir,'visualizations')):
        #Data
        dfs = NYISOVis.dfs_energy(year=year, f=f)
        
        #Plots
        fig, ax = plt.subplots(figsize=(10,5), dpi=300)
        dfs['fuel_mix'].plot.area(ax=ax,
                                  color=[LEGEND_DEETS.get(x, '#333333') for x in dfs['fuel_mix'].columns],
                                  alpha=0.9, lw=0) #fuel mix
        dfs['load'].plot.line(ax=ax, linestyle='dashed', linewidth=1, color='k', label='Load') #load line
        
        #Legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fancybox=True, shadow=False)
        #Axes
        if f == 'M':
            ylabel = 'Energy [TWh]'
        else:
            ylabel = 'Energy [GWh]'
        ax.set(title=f'Historic ({year})',
               xlabel=None, ylabel=ylabel,
               xlim=[dfs['fuel_mix'].index[0], dfs['fuel_mix'].index[-1]],
               ylim=None)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
        plt.setp(ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines(), markersize=3)
                
        #Save and show
        file = pl.Path(out_dir, f'{year}_energy_{f}.svg')
        fig.savefig(file, bbox_inches='tight', transparent=True)
        fig.show()

    @staticmethod
    def dfs_clcpa_carbon_free(year='2019', f='D'):
        dfs = NYISOVis.dfs_energy(year=year, f=f)
        #Calculating Carbon-free Fraction [%]
        dfs['ef'] = dfs['fuel_mix'].div(dfs['load'], axis='index') * 100
        dfs['ef']['percent_carbon_free'] = dfs['ef'][CARBONFREE_SOURCES].sum(axis='columns')
        dfs['df'] = dfs['ef'][CARBONFREE_SOURCES] #plot only carbon free resources
        return dfs
        
    @staticmethod
    def fig_clcpa_carbon_free(year='2019', f='D',
                              out_dir = pl.Path(c_dir,'visualizations')):
        """
        Figure Inspiration: NYISO Power Trends 2020 - Figure 12: Production of In-State Renewables & Zero-Emission Resources Relative to 2019 Load 
        """
        dfs = NYISOVis.dfs_clcpa_carbon_free(year=year, f=f) # Data
        
        #Plot Carbon-free Fraction
        fig, ax = plt.subplots(figsize=(10,5), dpi=300)
        dfs['df'].plot.area(ax=ax,
                            color=[LEGEND_DEETS.get(x, '#333333') for x in dfs['df'].columns],
                            alpha=0.9, lw=0)
       
        #Plot Import Line
        gen_imp = dfs['ef'][CARBONFREE_SOURCES+['Imports']].sum(axis='columns')
        gen_imp.index = gen_imp.index.astype('O')
        gen_imp.plot.line(ax=ax,linestyle='dotted',
                          linewidth=1, color='k', label='Total + Imports')
    
        #Plot Goals and Progress
        data = [[t for t in CARBONFREE_SOURCES if t!='Nuclear'],
                CARBONFREE_SOURCES]
        averages = ['Renewable: {:.0f}% + Imports: {:.0f}% (70% by 2030)',
                    'Carbon-Free: {:.0f}% + Imports: {:.0f}% (100% by 2040)']
        colors = ['limegreen','lawngreen']
        h_distances = [0.05, 0.05]
        for t,l,c,h in zip(data, averages, colors, h_distances):
            avg = dfs['fuel_mix'][t].sum(axis='index').sum() / dfs['load'].sum(axis='index') * 100
            avg_imp = dfs['fuel_mix'][t+['Imports']].sum(axis='index').sum() / dfs['load'].sum(axis='index') * 100
            ax.axhline(y=avg, xmax=h, color='k', linestyle='solid', lw=1)
            ax.text(h, avg/100, l.format(avg, avg_imp), 
                     bbox=dict(boxstyle='round',ec='black',fc=c, alpha=0.9),
                     transform=ax.transAxes)
            
        #Legend
        ax.legend(loc='upper center',bbox_to_anchor=(0.45, -0.05),
                  ncol=5, fancybox=True, shadow=False)
        #Axes
        ax.set(title=f'Historic ({year})',
               xlabel=None, ylabel='Percent of Load Served by Carbon-Free Energy',
               xlim=[dfs['df'].index[0], dfs['df'].index[-1]], ylim=[0, 100])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
        plt.setp(ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines(), markersize=3)
        
        #Save and show
        file = pl.Path(out_dir, f'{year}_clcpa_carbon_free_{f}.svg')
        fig.savefig(file, bbox_inches='tight', transparent=True)
        fig.show()
            
    @staticmethod
    def fig_carbon_free_year(year='2019', out_dir = pl.Path(c_dir,'visualizations')):
        stats = NYISOStat.table_annual_energy(year=year)
        df = stats[f'Historic ({year}) [% of Load]'].drop(index=['Total Renewable Generation',
                                                                 'Total Carbon-Free Generation',
                                                                 'Total Generation',
                                                                 'Total Generation + Imports',
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
                    label='Carbon-Free Generation')
        ax.text(-0.575, perc,'{:.0f}'.format(perc))
        # Carbon Free + Imports Line
        perc = stats[f'Historic ({year}) [% of Load]'].loc[['Total Carbon-Free Generation',
                                                            'Imports']].sum()
        ax.axhline(y=perc, color='k', linestyle='dotted',
                    label='Carbon-Free Generation + Imports')
        ax.text(-0.575, perc,'{:.0f}'.format(perc))
        
        #Legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels),
                  loc='right', bbox_to_anchor=(2.2, 0.5),
                  ncol=1, fancybox=True, shadow=False)
        #Axes
        ax.set(title=f'Historic ({year})',
               xlabel=year, ylabel='Percent of Load Served by Carbon-Free Energy',
               xlim=None, ylim=None)
        plt.xticks([])
        fig.savefig(pl.Path(out_dir,f'{year}_carbon_free_year.png'))
        fig.show()

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
