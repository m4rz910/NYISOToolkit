import pathlib as pl

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import yaml

from nyisotoolkit.nyisodata.nyisodata import NYISOData
from nyisotoolkit.nyisostat.nyisostat import NYISOStat

# Figure Configuration
plt.style.use(['seaborn-whitegrid'])
plt.rcParams.update({'font.weight': 'bold',
                     'font.size': 8,
                     'axes.labelweight': 'bold',
                     'lines.linewidth': 2,
                     'axes.titleweight': 'bold',
                     'axes.grid': True,
                     'axes.edgecolor': '.15'})

# Legend Colors
c_dir = pl.Path(__file__).resolve().parent
figure_config_path = pl.Path(c_dir, 'legend_colors.yml')
with open(figure_config_path) as file:
    all_configs = yaml.load(file, Loader=yaml.FullLoader)
    LEGEND_DEETS = all_configs['legend_colors']
    
# List of Carbon Free Resources
CARBONFREE_SOURCES = ['Hydro','Other Renewables','Wind','Nuclear']


class NYISOVis:
    """A class used to create power system visualizations from the NYISOData and NYISOStat modules
    
    Attributes
    ----------
    year: str, int
        Year of dataset(s) to use for graphs
    out_dir: Pathlib Object, optional
        Path to a directory to save the figures to in SVG format (default location is in nyisotoolkit/nyisovis/visualizations)
        
    Methods
    -------
    tables_energy
        Gathers datasets needed to produce fig_energy
    fig_energy
        Produces a stacked area chart showing the trend of energy generation by energy source from tables_energy
    tables_clcpa_carbon_free
        Gathers datasets needed to produce fig_clcpa_carbon_free
    fig_clcpa_carbon_free
        Produces a stacked area chart showing the trend of carbon-free that served load by energy source from tables_clcpa_carbon_free
    fig_carbon_free_year
        Produces a stacked bar chart of annual percent carbon-free by energy source
    """
    
    def __init__(self, year='2019', out_dir = pl.Path(c_dir,'visualizations')):
        """
        Parameters
        ----------
        year: str, int
            Year of dataset(s) to use for graphs
        """
        self.year = year
        self.out_dir = out_dir

    @staticmethod
    def tables_energy(year, f='D'):
        """Gathers datasets needed to produce fig_energy
        
        Parameters
        ----------
        f: str
            Frequency of graph to generate (daily ('D') and monthly ('M') recommended)
        
        Returns
        -------
        tables: Dictionary
            Dictionary containing dataset names as keys and respective Dataframes
        """
        
        if f in ['Y']:
            raise ValueError('Frequency Not Supported!')
        
        #Power [MW]
        load = NYISOData(dataset='load_5m', year=year).df.tz_convert('US/Eastern')['NYCA']
        fuel_mix = NYISOData(dataset='fuel_mix_5m', year=year).df.tz_convert('US/Eastern')
        imports = NYISOData(dataset='interface_flows_5m', year=year).df.tz_convert('US/Eastern')
        imports = imports.loc[:, ('External Flows', slice(None), 'Flow (MW)')]
        imports.drop(('External Flows', 'SCH - HQ IMPORT EXPORT', 'Flow (MW)'),
                     axis='columns', inplace=True) #'SCH - HQ IMPORT EXPORT' is a subset of another external flow
        imports = imports.sum(axis='columns')
        
        tables = {'load':load, 'fuel_mix':fuel_mix, 'imports': imports} # group datasets into dictionary to apply 
        def power_to_energy(df):
            """Energy Converstion [MWh] and Resampling By Summing Energy"""
            return (df * 1/12).resample(f).sum()/1000  #MW->MWh->GWh
        tables = {k: power_to_energy(v) for k, v in tables.items()}
        
        tables['fuel_mix']['Imports'] = tables['imports'] #add imports to fuel mix
        del tables['imports'] #remove imports from the tables list
        
        order = ['Nuclear','Hydro','Other Renewables','Wind','Natural Gas','Dual Fuel',
                 'Other Fossil Fuels', 'Imports'] 
        tables['fuel_mix'] = tables['fuel_mix'][order] #fix order
        
        def month_adj_object(df):
            """Adjust index for months and make index objects to label correctly"""
            if f == 'M':
                df.index = df.index.shift(-1,'M').shift(1,'D')
                df = df/1000 # GWh->TWh
            df.index = df.index.astype('O')
            return df       
        tables = {k: month_adj_object(v) for k, v in tables.items()}
        return tables
        
    def fig_energy(self, f):
        """Produces a stacked area chart showing the trend of energy generation by energy source from tables_energy
        
        Parameters
        ----------
        f: str
            Frequency of graph to generate (daily ('D') and monthly ('M') recommended)
        """
        
        tables = self.tables_energy(year=self.year, f=f) #Data
        #Plots
        fig, ax = plt.subplots(figsize=(6,3), dpi=300)
        tables['fuel_mix'].plot.area(ax=ax,
                                  color=[LEGEND_DEETS.get(x, '#333333') for x in tables['fuel_mix'].columns],
                                  alpha=0.9, lw=0) #fuel mix
        tables['load'].plot.line(ax=ax, linestyle='dashed', linewidth=1, color='k', label='Load') #load line
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fancybox=True, shadow=False) #Legend
        #Axes
        if f == 'M':
            ylabel = 'Energy [TWh]'
        else:
            ylabel = 'Energy [GWh]'
        ax.set(title=f'Historic ({self.year})',
               xlabel=None, ylabel=ylabel,
               xlim=[tables['fuel_mix'].index[0], tables['fuel_mix'].index[-1]],
               ylim=None)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
        plt.setp(ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines(), markersize=3)
        # NYISOToolkit label and source
        ax.text(0.875, 0.015, 'NYISOToolkit (Datasource: NYISO OASIS)',
                c='black', fontsize='4', fontstyle= 'italic', horizontalalignment='center',
                alpha=0.6, transform=ax.transAxes)
        
        #Save and show
        file = pl.Path(self.out_dir, f'{self.year}_energy_{f}.png')
        fig.savefig(file, bbox_inches='tight', transparent=True)
        # fig.show()

    @staticmethod
    def tables_clcpa_carbon_free(year, f='D'):
        """Gathers datasets needed to produce fig_clcpa_carbon_free
        
        Parameters
        ----------
        f: str
            Frequency of graph to generate (daily ('D') and monthly ('M') recommended)
            
        Returns
        -------
        tables: Dictionary
            Dictionary containing dataset names as keys and respective Dataframes
        """
        
        tables = NYISOVis.tables_energy(year=year, f=f)
        #Calculating Carbon-free Fraction [%]
        tables['ef'] = tables['fuel_mix'].div(tables['load'], axis='index') * 100
        tables['ef']['percent_carbon_free'] = tables['ef'][CARBONFREE_SOURCES].sum(axis='columns')
        tables['df'] = tables['ef'][CARBONFREE_SOURCES] #plot only carbon free resources
        return tables
        
    def fig_clcpa_carbon_free(self, f='D'):
        """Produces a stacked area chart showing the trend of carbon-free that served load by energy source from tables_clcpa_carbon_free
        
        Figure Inspiration: NYISO Power Trends 2020 - Figure 12: Production of In-State Renewables & Zero-Emission Resources Relative to 2019 Load 
        
        Parameters
        ----------
        f: str
            Frequency of graph to generate (daily ('D') and monthly ('M') recommended)
        
        """
        tables = self.tables_clcpa_carbon_free(year=self.year, f=f) # Data
        
        #Plot Carbon-free Fraction
        fig, ax = plt.subplots(figsize=(6,3), dpi=300)
        tables['df'].plot.area(ax=ax,
                            color=[LEGEND_DEETS.get(x, '#333333') for x in tables['df'].columns],
                            alpha=0.9, lw=0)
       
        #Plot Import Line
        gen_imp = tables['ef'][CARBONFREE_SOURCES+['Imports']].sum(axis='columns')
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
            avg = tables['fuel_mix'][t].sum(axis='index').sum() / tables['load'].sum(axis='index') * 100
            avg_imp = tables['fuel_mix'][t+['Imports']].sum(axis='index').sum() / tables['load'].sum(axis='index') * 100
            ax.axhline(y=avg, xmax=h, color='k', linestyle='solid', lw=1)
            ax.text(h, avg/100, l.format(avg, avg_imp), 
                     bbox=dict(boxstyle='round',ec='black',fc=c, alpha=0.9),
                     transform=ax.transAxes)
        # NYISOToolkit label and source
        ax.text(0.875, 0.015, 'NYISOToolkit (Datasource: NYISO OASIS)',
                c='black', fontsize='4', fontstyle= 'italic', horizontalalignment='center',
                alpha=0.6, transform=ax.transAxes)
            
        #Legend
        ax.legend(loc='upper center',bbox_to_anchor=(0.45, -0.05),
                  ncol=5, fancybox=True, shadow=False)
        #Axes
        ax.set(title=f'Historic ({self.year})',
               xlabel=None, ylabel='% of Demand Served by Carbon-Free Energy',
               xlim=[tables['df'].index[0], tables['df'].index[-1]], ylim=[0, 100])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
        plt.setp(ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines(), markersize=3)
        
        #Save and show
        file = pl.Path(self.out_dir, f'{self.year}_clcpa_carbon_free_{f}.png')
        fig.savefig(file, bbox_inches='tight', transparent=True)
        # fig.show()
            
    def fig_carbon_free_year(self):
        """Produces a stacked bar chart of annual percent carbon-free by energy source"""
        stats = NYISOStat(year=self.year).table_annual_energy()
        df = stats[f'Historic ({self.year}) [% of Load]'].drop(index=['Total Renewable Generation',
                                                                 'Total Carbon-Free Generation',
                                                                 'Total Generation',
                                                                 'Total Generation + Imports',
                                                                 'Load'])
        df = df.to_frame().T
        #Plot
        fig, ax = plt.subplots(figsize=(3,6), dpi=300)
        df.plot.bar(stacked=True, ax=ax,
                    color=[LEGEND_DEETS.get(x, '#333333') for x in df.columns],
                    alpha=0.9)
            
        # Renewable and Carbon Free lines with imports
        bar_left_edge=0.25
        bar_right_edge=0.75
        text_down_shift=0.9
        text_left_shift=0.08
        text_right_shift=0.03
        scatter_left= 0.27
        for quantity, c in zip(['Total Renewable Generation', 'Total Carbon-Free Generation'],
                               ['limegreen', 'lawngreen']):
            perc = stats[f'Historic ({self.year}) [% of Load]'].loc[quantity]
            ax.axhline(y=perc, xmin=bar_left_edge, xmax=bar_right_edge,
                       color=c, linestyle='dashed', linewidth=1, label=f'{quantity}')
            ax.text(-scatter_left-text_left_shift, perc-text_down_shift,'{:.0f}'.format(perc))
            ax.scatter(-scatter_left, perc, marker='>', color=c, alpha=1)
            
            # With Imports Line
            perc = stats[f'Historic ({self.year}) [% of Load]'].loc[[quantity,'Imports']].sum()
            ax.axhline(y=perc, xmin=bar_left_edge, xmax=bar_right_edge,
                       color=c, linestyle='dotted', linewidth=1, label=f'{quantity} + Imports')
            ax.text(scatter_left+text_right_shift, perc-text_down_shift,'{:.0f}'.format(perc))
            ax.scatter(scatter_left, perc, marker='<', color=c, alpha=1)
        
        # NYISOToolkit label and source
        ax.text(1.3, 0, 'NYISOToolkit (Datasource: NYISO OASIS)',
                c='black', fontsize='4', fontstyle= 'italic', horizontalalignment='center',
                alpha=0.6, transform=ax.transAxes)
        
        #Legend
        def remove_text(x):
            return x.replace('Total','').replace('Generation','')
        handles, labels = ax.get_legend_handles_labels()
        labels = [remove_text(l) for l in labels]
        ax.legend(reversed(handles), reversed(labels),
                  loc='right', bbox_to_anchor=(2, 0.5),
                  ncol=1, fancybox=True, shadow=False)
        
        #Axes
        ax.set(title=f'Historic ({self.year})',
               xlabel=self.year, ylabel='% of Demand Served by Carbon-Free Energy',
               xlim=None, ylim=None)
        plt.xticks([])
        
        #Save
        file = pl.Path(self.out_dir,f'{self.year}_carbon_free_year.png')
        fig.savefig(file, bbox_inches='tight', transparent=True)
        # fig.show()        
        
    def fig_carbon_free_years():
        """Todo: stacked area chart over time using nyisostat annual summary"""
        return
    
    def net_load():
        """Todo: Load vs Net Load Shape"""
        return