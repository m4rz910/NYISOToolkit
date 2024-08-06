import pathlib as pl
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import yaml
import pandas as pd
import numpy as np

from nyisotoolkit.nyisodata.nyisodata import NYISOData, DATABASE_DIR, table_load_weighted_price
from nyisotoolkit.nyisostat.nyisostat import NYISOStat

# Legend Colors
c_dir = pl.Path(__file__).resolve().parent
figure_config_path = pl.Path(c_dir, "legend_colors.yml")
with open(figure_config_path) as file:
    all_configs = yaml.load(file, Loader=yaml.FullLoader)
    LEGEND_DEETS = all_configs["legend_colors"]
    
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
    tables_energy_generation
        Gathers datasets needed to produce fig_energy
    fig_energy_generation
        Produces a stacked area chart showing the trend of energy generation by energy source from tables_energy
    tables_carbon_free_timeseries
        Gathers datasets needed to produce fig_carbon_free_timeseries
    fig_carbon_free_timeseries
        Produces a stacked area chart showing the trend of carbon-free that served load by energy source from tables_clcpa_carbon_free
    fig_carbon_free_year
        Produces a stacked bar chart of annual percent carbon-free by energy source
        
    todo: Add additional functions
    """
    
    def __init__(self, year='2019', out_dir = pl.Path(c_dir,'visualizations'), **kwargs):
        """
        Parameters
        ----------
        year: str, int
            Year of dataset(s) to use for graphs
        out_dir: Pathlib Object
            Directory to save the figure to
        kwargs:
            Additional parameters to be passed to NYISOData initialization
        """
        
        self.year = year
        self.out_dir = out_dir
        
        #if redownload is passed, update existing databases
        #looks to see which datasets are available and redownloads them
        if kwargs.get("redownload", False) or kwargs.get("reconstruct", False):
            existing_dbs = pl.Path(DATABASE_DIR).glob(f"{self.year}*.pkl")
            existing_datasets = [db.name.replace(".pkl","").replace(f"{self.year}_","") for db in existing_dbs]
            for dataset in existing_datasets:
                NYISOData(dataset=dataset, year=self.year, **kwargs)

    def tables_energy_generation(self, f='D'):
        """Gathers datasets (in US/Eastern) needed to produce a few figures.
        
        Parameters
        ----------
        f: str
            Frequency of graph to generate (daily ('D') and monthly ("ME") recommended)

        Returns
        -------
        tables: Dictionary
            Dictionary containing dataset names as keys and respective Dataframes
        """
        
        if f in ['Y']:
            raise ValueError('Frequency Not Supported!')
        
        #Power [MW]
        load = NYISOData(dataset='load_5m', year=self.year).df.tz_convert('US/Eastern')['NYCA']
        fuel_mix = NYISOData(dataset='fuel_mix_5m', year=self.year).df.tz_convert('US/Eastern')
        imports = NYISOData(dataset='interface_flows_5m', year=self.year).df.tz_convert('US/Eastern')
        imports = imports.loc[:, ('External Flows', slice(None), 'Flow (MW)')]
        imports.drop(('External Flows', 'SCH - HQ IMPORT EXPORT', 'Flow (MW)'),
                     axis='columns', inplace=True) #'SCH - HQ IMPORT EXPORT' is a subset of another external flow
        imports = imports.sum(axis='columns')
        
        tables = {'load':load, 'fuel_mix':fuel_mix, 'imports': imports} # group datasets into dictionary to apply 
        def power_to_energy(df):
            """Energy Converstion [MWh] and Resampling By Summing Energy"""
            return (df * 1 / 12).resample(f).sum() / 1000  # MW->MWh->GWh

        tables = {k: power_to_energy(v) for k, v in tables.items()}

        tables["fuel_mix"]["Imports"] = tables["imports"]  # add imports to fuel mix
        del tables["imports"]  # remove imports from the tables list

        order = [
            "Nuclear",
            "Hydro",
            "Other Renewables",
            "Wind",
            "Natural Gas",
            "Dual Fuel",
            "Other Fossil Fuels",
            "Imports",
        ]
        tables["fuel_mix"] = tables["fuel_mix"][order]  # fix order

        def month_adj_object(df):
            """Adjust index for months and make index objects to label correctly"""
            if f == "ME":
                df.index = df.index.shift(-1, "ME").shift(1, "D")
                df = df / 1000  # GWh->TWh
            df.index = df.index.astype("O")
            return df

        tables = {k: month_adj_object(v) for k, v in tables.items()}
        return tables
        
    def fig_energy_generation(self, f='D'):
        """Produces a stacked area chart showing the trend of energy generation by energy source from tables_energy

        Parameters
        ----------
        f: str
            Frequency of graph to generate (daily ('D') and monthly ("ME") recommended)
        """
        
        tables = self.tables_energy_generation(f=f) #Data
        #Plots
        fig, ax = plt.subplots(figsize=(6,3), dpi=300)
        tables['fuel_mix'].plot.area(ax=ax,
                                  color=[LEGEND_DEETS.get(x, '#333333') for x in tables['fuel_mix'].columns],
                                  alpha=0.9, lw=0) #fuel mix
        tables['load'].plot.line(ax=ax, linestyle='dashed', linewidth=1, color='k', label='Load') #load line
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fancybox=True, shadow=False) #Legend
        #Axes
        if f == "ME":
            ylabel = 'Energy [TWh]'
        else:
            ylabel = 'Energy [GWh]'
        ax.set(title=f'Energy Generation ({self.year})',
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
        
        #Save
        file = pl.Path(self.out_dir, f'{self.year}_energy_generation_{f}.png')
        fig.savefig(file, bbox_inches='tight', transparent=False)

    def tables_carbon_free_timeseries(self, f='D'):
        """Gathers datasets needed to produce fig_carbon_free_timeseries
        
        Parameters
        ----------
        f: str
            Frequency of graph to generate (daily ('D') and monthly ("ME") recommended)

        Returns
        -------
        tables: Dictionary
            Dictionary containing dataset names as keys and respective Dataframes
        """
        
        tables = self.tables_energy_generation(f=f) #Calculating Carbon-free Fraction [%]
        tables['ef'] = tables['fuel_mix'].div(tables['load'], axis='index') * 100
        tables['ef']['percent_carbon_free'] = tables['ef'][CARBONFREE_SOURCES].sum(axis='columns')
        tables['df'] = tables['ef'][CARBONFREE_SOURCES] #only carbon free resources
        return tables
        
    def fig_carbon_free_timeseries(self, f='D'):
        """Produces a stacked area chart showing the trend of carbon-free that served load by energy source from tables_carbon_free_timeseries. Overlayed on top are the CLCPA targets and the status toward achieving them. 
        
        Figure Inspiration: NYISO Power Trends 2020 - Figure 12: Production of In-State Renewables & Zero-Emission Resources Relative to 2019 Load 
        
        Parameters
        ----------
        f: str
            Frequency of graph to generate (daily ('D') and monthly ("ME") recommended)

        """
        tables = self.tables_carbon_free_timeseries(f=f) # Data
        
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
        for t, l, c, h in zip(data, averages, colors, h_distances):
            avg = (
                tables["fuel_mix"][t].sum(axis="index").sum()
                / tables["load"].sum(axis="index")
                * 100
            )
            avg_imp = (
                tables["fuel_mix"][t + ["Imports"]].sum(axis="index").sum()
                / tables["load"].sum(axis="index")
                * 100
            )
            ax.axhline(y=avg, xmax=h, color="k", linestyle="solid", lw=1)
            ax.text(
                h,
                avg / 100,
                l.format(avg, avg_imp),
                bbox=dict(boxstyle="round", ec="black", fc=c, alpha=0.9),
                transform=ax.transAxes,
            )
        # NYISOToolkit label and source

        ax.text(0.875, 0.015, 'NYISOToolkit (Datasource: NYISO OASIS)',
                c='black', fontsize='4', fontstyle= 'italic', horizontalalignment='center',
                alpha=0.6, transform=ax.transAxes)
            
        #Legend
        ax.legend(loc='upper center',bbox_to_anchor=(0.45, -0.05),
                  ncol=5, fancybox=True, shadow=False)
        #Axes
        ax.set(title=f'Carbon-free Time Series ({self.year})',
               xlabel=None, ylabel='% of Demand Served by Carbon-Free Energy',
               xlim=[tables['df'].index[0], tables['df'].index[-1]], ylim=[0, 100])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
        plt.setp(ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines(), markersize=3)
        
        #Save
        file = pl.Path(self.out_dir, f'{self.year}_carbon_free_timeseries_{f}.png')
        fig.savefig(file, bbox_inches='tight', transparent=False)
        return tables
            
    def fig_carbon_free_year(self):
        """Produces a stacked bar chart of annual percent carbon-free by energy source"""
        stats = NYISOStat(year=self.year).table_annual_energy()
        df = stats[f"Historic ({self.year}) [% of Load]"].drop(
            index=[
                "Total Renewable Generation",
                "Total Carbon-Free Generation",
                "Total Generation",
                "Total Generation + Imports",
                "Load",
            ]
        )
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
        ax.set(title=f'Carbon-free Year ({self.year})',
               xlabel=self.year, ylabel='% of Demand Served by Carbon-Free Energy',
               xlim=None, ylim=None)
        plt.xticks([])
        
        #Save
        file = pl.Path(self.out_dir,f'{self.year}_carbon_free_year.png')
        fig.savefig(file, bbox_inches='tight', transparent=False)
        
    def fig_decarbonization_heatmap(self):
        """Creates a figure depicting an overview of the seasonal and daily carbon-free operation.
        Inspired by: Google ESIG Presentation 10/13/2020
        """
        
        tables = self.tables_carbon_free_timeseries(f="h") # Data
        df = pd.DataFrame(tables['df'].sum(axis='columns')) #sum to get total carbon-free percent'
        df.index = pd.to_datetime(df.index)
        df['Date'] = df.index.date
        df['Hour'] = df.index.hour
        df = df.pivot_table(index='Hour', columns='Date', values=0) #TODO: Check: duplicate index warning with regular pivot, likely because in EST TIME.
        
        #Plot
        fig, ax = plt.subplots(figsize=(6,3), dpi=300)
        cmap = sns.diverging_palette(2, 145, as_cmap=True)
        ax = sns.heatmap(df, cmap=cmap, vmin=0, vmax=100, ax=ax)
        ax.collections[0].colorbar.ax.set_ylabel('% of Demand Served by Carbon-Free Energy',
                                                 rotation=270, labelpad=10)
        
        #Axes
        ax.set(title=f'Decarbonization Heat Map ({self.year})',
               xlabel=None, ylabel=None,
               xlim=[0,364], ylim=None)
        #x-axes
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
        #y-axes
        ax.set_yticks(range(6,24,6))
        ax.set_yticklabels(['Morning (6AM)', 'Noon (12PM)', 'Evening (6PM)'])
        plt.setp(ax.get_yticklabels(), rotation=0,
                 horizontalalignment='right')
        #both axes
        plt.setp(ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines(), markersize=3)
        
        # NYISOToolkit label and source
        ax.text(0.845, 0.015, 'NYISOToolkit (Datasource: NYISO OASIS)',
                c='black', fontsize='4', fontstyle= 'italic', horizontalalignment='center',
                alpha=0.6, transform=ax.transAxes)
                
        #Save
        file = pl.Path(self.out_dir,f'{self.year}_decarbonization_heatmap.png')
        fig.savefig(file, bbox_inches='tight', transparent=False)

    def fig_decarbonization_clock(self):
        """Creates a figure depicting 24 hour clock of the carbon-free operation of the average day.
        Inspired by: Google ESIG Presentation 10/13/2020
        """
        tables = self.tables_carbon_free_timeseries(f="h") # Data
        df = pd.DataFrame(tables['df'].sum(axis='columns')) #sum for total carbon-free %
        df.index = pd.to_datetime(df.index)
        df = df.groupby(df.index.hour).mean().sort_index(ascending=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(6,6), dpi=300)
        cmap = sns.diverging_palette(2, 145, as_cmap=True)
        wedges, texts = ax.pie(np.ones(df.shape[0]),
                               startangle=90+(360/24/2),
                               colors=[c[0][:3] for c in cmap(df.values/100)],
                               wedgeprops=dict(width=0.5)
                               )
        
        # Colorbar
        cax, cbar_kwds = mpl.colorbar.make_axes(ax, shrink=0.50, location="right", pad=-0.05)
        cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, orientation='vertical')
        cb1.ax.set_yticklabels([i for i in range(0,120,20)])
        cb1.ax.set_ylabel('% of Demand Served by Carbon-Free Energy', rotation=270, labelpad=10)
        cb1.outline.set_visible(False)
        
        # Annotations
        hours = df.index
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            r = 0.55
            y = np.sqrt(r)*np.sin(np.deg2rad(ang))
            x = np.sqrt(r)*np.cos(np.deg2rad(ang))
            ax.annotate(hours[i], xy=(x, y),
                        horizontalalignment="center",
                        verticalalignment="center"
                        )
        
        # NYISOToolkit label and source
        ax.text(0.845, 0.08, 'NYISOToolkit (Datasource: NYISO OASIS)',
                c='black', fontsize='4', fontstyle= 'italic', horizontalalignment='center',
                alpha=0.6, transform=ax.transAxes)
    
        # axes
        ax.set_title(f'Decarbonization Clock ({self.year})', y=0.91)
        
        # Save
        file = pl.Path(self.out_dir,f'{self.year}_decarbonization_clock.png')
        fig.savefig(file, bbox_inches='tight', transparent=False)
    
    def fig_demand_pdf(self, cumulative=False):
        """Creates a figure of the year's demand probability density function."""
        df = NYISOData(dataset="load_5m", year=self.year).df.tz_convert('US/Eastern')["NYCA"]/1000 #MW->GW
        df = pd.DataFrame(df)
        ax_kwargs = {"title":f'State-wide Demand Probability Distribution ({self.year})',
                     "xlabel":"Demand [GW]", "ylabel":"Probability [%]",
                     "xlim":(df.values.min(),df.values.max()), "ylim":None}
        
        HISTPLOT_KWARGS.update({"cumulative":cumulative})
        if cumulative:
            ax_kwargs.update({"title": f'State-wide Demand Cumulative Probability Distribution ({self.year})'})
        fig = pdf(df.rename(columns={'NYCA':"Values"}), ax_kwargs, HISTPLOT_KWARGS)
        # Save
        if cumulative:
            file = pl.Path(self.out_dir,f'{self.year}_demand_cumulative_pdf.png')
        else:
            file = pl.Path(self.out_dir,f'{self.year}_demand_pdf.png')
        fig.savefig(file, bbox_inches='tight', transparent=False)
    
    def fig_demand_forecast_error(self, cumulative=False):
        load = NYISOData(dataset="load_h", year=self.year).df.tz_convert('US/Eastern')["NYCA"] #MW
        load_forecast = NYISOData(dataset="load_forecast_h", year=self.year).df.tz_convert('US/Eastern')["NYCA"] #MW
        load_forecast = load_forecast.loc[load.index]
        df = (load_forecast-load)/load *100 # error [%]
        df = pd.DataFrame(df)
        ax_kwargs = {"title":f'State-wide Demand Forecast Error Probability Distribution ({self.year})',
                     "xlabel":"Demand [MW]", "ylabel":"Probability [%]",
                     "xlim":(df.values.min(),df.values.max()), "ylim":None}
        
        HISTPLOT_KWARGS.update({"cumulative":cumulative})
        if cumulative:
            ax_kwargs.update({"title": f'State-wide Demand Forecast Error Cumulative Probability Distribution ({self.year})'})
        fig = pdf(df.rename(columns={'NYCA':"Values"}), ax_kwargs, HISTPLOT_KWARGS)
        # Save
        if cumulative:
            file = pl.Path(self.out_dir,f'{self.year}_demand_forecast_error_cumulative_pdf.png')
        else:
            file = pl.Path(self.out_dir,f'{self.year}_demand_forecast_error_pdf.png')
        fig.savefig(file, bbox_inches='tight', transparent=False)

    def fig_price_pdf(self, rt, cumulative = False):
        """Creates a figure of the state-wide average energy price probability distribution.
        """
        df = table_load_weighted_price(year=self.year, rt=rt) #$/MWh
        da_rt = "RT" if rt else "DA"
        ax_kwargs = {"title":f'State-wide Average Baseload {da_rt} Energy Price\n Probability Distribution ({self.year})',
                     "xlabel":"Price ($/MWh)", "ylabel":"Probability [%]",
                     "xlim":(df.values.min(),df.values.max()), "ylim":None}
        HISTPLOT_KWARGS.update({"cumulative":cumulative})
        if cumulative:
            ax_kwargs.update({"title": f'State-wide Average Baseload {da_rt} Energy Price\n Cumulative Probability Distribution ({self.year})'})
        fig = pdf(df.rename(columns={0:"Values"}), ax_kwargs, HISTPLOT_KWARGS)
        # Save
        if cumulative:
            file = pl.Path(self.out_dir,f'{self.year}_{da_rt}_price_cumulative_pdf.png')
        else:
            file = pl.Path(self.out_dir,f'{self.year}_{da_rt}_price_pdf.png')
        fig.savefig(file, bbox_inches='tight', transparent=False)
        
    def fig_price_difference_pdf(self, cumulative = False):
        rt_price = table_load_weighted_price(year=self.year, rt=True).resample("h").mean() #$/MWh
        da_price = table_load_weighted_price(year=self.year, rt=False) #$/MWh
        df = da_price - rt_price
        ax_kwargs = {"title":f'State-wide Average Baseload DA-RT Energy Price Difference\n Probability Distribution ({self.year})',
                     "xlabel":"Price ($/MWh)", "ylabel":"Probability [%]",
                     "xlim":(df.values.min(),df.values.max()), "ylim":None}
        
        HISTPLOT_KWARGS.update({"cumulative":cumulative})
        if cumulative:
            ax_kwargs.update({"title": f'State-wide Average Baseload DA-RT Energy Price Difference \n Cumulative Probability Distribution ({self.year})'})
        fig = pdf(df.rename(columns={0:"Values"}), ax_kwargs, HISTPLOT_KWARGS)
        # Save
        if cumulative:
            file = pl.Path(self.out_dir,f'{self.year}_price_difference_cumulative_pdf.png')
        else:
            file = pl.Path(self.out_dir,f'{self.year}_price_difference_pdf.png')
        fig.savefig(file, bbox_inches='tight', transparent=False)
    
    def fig_carbon_free_years(self):
        """Todo: stacked area chart over time using nyisostat annual summary"""
        return

    def net_load(self):
        """Todo: Load vs Net Load Shape"""
        return
    
def pdf(df, ax_kwargs, histplot_kwargs):
    """General function for generating probabilty density functions."""
    fig, ax = plt.subplots(figsize=(6,3), dpi=300)
    df['Season'] = df.index.month.map(SEASONS)
    histplot_kwargs.update({"data": df, "x": "Values",
                            "hue": 'Season',"palette": SEASON_COLORS,
                            "ax": ax,
                            "stat": "probability",
                            "kde": True,
                            "alpha": 0.5,
                            "legend":True,
                            })
    ax = sns.histplot(**histplot_kwargs)
    ax = fix_legend(ax, loc='right', bbox_to_anchor=(1.2, 0.5), ncol=1)
    ax.set(**ax_kwargs)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, symbol=None))

    # NYISOToolkit label and source
    ax.text(0.87, 0.01, 'NYISOToolkit (Datasource: NYISO OASIS)',
            c='black', fontsize='4', fontstyle= 'italic', horizontalalignment='center',
            alpha=0.6, transform=ax.transAxes)
    return ax.figure

def fix_legend(ax, **kws):
    """https://github.com/mwaskom/seaborn/issues/2280"""
    old_legend = ax.legend_
    handles = old_legend.legend_handles
    labels = [t.get_text() for t in old_legend.get_texts()]
    #title = old_legend.get_title().get_text()
    ax.legend(handles, labels, **kws)
    return ax

def basic_plots(nyisovis_kwa):
    nv = NYISOVis(**nyisovis_kwa)
    nv.fig_carbon_free_year()
    nv.fig_decarbonization_heatmap()
    nv.fig_decarbonization_clock()
    for f in ['D',"ME"]:
        nv.fig_energy_generation(f=f)
        nv.fig_carbon_free_timeseries(f=f)
            
def statistical_plots(nyisovis_kwa):
    nv = NYISOVis(**nyisovis_kwa)
    for rtorda in [False, True]:
        for c in [False, True]:
            nv.fig_demand_pdf(cumulative=c)
            nv.fig_demand_forecast_error(cumulative=c)
            nv.fig_price_pdf(rt=rtorda, cumulative=c)
            nv.fig_price_difference_pdf(cumulative=c)

# Figure Configuration
plt.style.use(['seaborn-v0_8-whitegrid'])
plt.rcParams.update({'font.weight': 'bold',
                     'font.size': 8,
                     'axes.labelweight': 'bold',
                     'lines.linewidth': 2,
                     'axes.titleweight': 'bold',
                     'axes.grid': True,
                     'axes.edgecolor': '.15',
                     'legend.frameon': False})

# List of Carbon Free Resources
CARBONFREE_SOURCES = ["Hydro", "Other Renewables", "Wind", "Nuclear"]

SEASONS = {12:"Winter", 1:"Winter", 2: "Winter",
           3:"Spring", 4:"Spring", 5:"Spring",
           6:"Summer", 7:"Summer", 8:"Summer",
           9:"Fall", 10:"Fall", 11:"Fall"
          }

SEASON_COLORS = {"Winter": "tab:gray",
                 "Spring": "tab:green",
                 "Summer": "tab:cyan",
                 "Fall"  : "tab:orange"
                 }

HISTPLOT_KWARGS = {"data": None,
                   "ax": None,
                   "x": None, "y": None, 
                   "hue": None, "weights": None,
                   "stat": 'count',
                   "bins": 'auto', "binwidth": None, "binrange": None,
                   "discrete": None,
                   "cumulative": False,
                   "common_bins": True, "common_norm": True,
                   "multiple": 'layer', "element": 'step', 
                   "fill": True, "shrink": 1, 
                   "kde": False, "kde_kws": None, "line_kws": None, 
                   "thresh": 0, "pthresh": None, "pmax": None,
                   "cbar": False, "cbar_ax": None, "cbar_kws": None,
                   "palette": None, "hue_order": None, "hue_norm": None, "color": None, 
                   "log_scale": None,
                   "legend": True}