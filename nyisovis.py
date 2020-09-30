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
    def fig_clcpa_carbon_free(year='2019', f='D',
                              out_dir = pl.Path(c_dir,'visualizations')):
        """
        Figure Inspiration: NYISO Power Trends 2020 - Figure 12: Production of In-State Renewables & Zero-Emission Resources Relative to 2019 Load 
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
        fuel_mix['Net Imports'] = imports #Add Imports to fuel mix
    
        #Calculating Carbon-free Fraction [%]
        ef = fuel_mix.div(load, axis='index') * 100
        carbonfree_sources = ['Hydro','Other Renewables','Wind','Nuclear']
        ef['percent_carbon_free'] = ef[carbonfree_sources].sum(axis='columns')
                    
        #Plot Carbon-free Fraction
        fig, ax = plt.subplots(figsize=(10,5), dpi=300)
        plt.title(f'Historic ({year})')
        df = ef[carbonfree_sources] #plot only carbon free resources
        df.plot.area(ax=ax,
                     color=[LEGEND_DEETS.get(x, '#333333') for x in df.columns],
                     alpha=0.9, lw=0)
        plt.ylabel('% of Load Served by NY CO$_2$e-Free Generation')
        
        #Plot Import Line
        gen_imp = ef[carbonfree_sources+['Net Imports']].sum(axis='columns')
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
            plt.axhline(y=avg, xmax=h, color='k', linestyle='solid', lw=1)
            plt.text(h, avg/100, l.format(avg, avg_imp), 
                     bbox=dict(boxstyle='round',ec='black',fc=c, alpha=0.9),
                     transform=ax.transAxes)
        #Legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(loc='lower center',bbox_to_anchor=(0.45, -0.15),
                  ncol=5, fancybox=True, shadow=False)
        #Axes
        plt.ylim(0,100)
        plt.xlim(ef.index[0], ef.index[-1])
        plt.xlabel('')
        plt.ylabel('Percent of Load Served by Carbon-Free Energy')
        
        locator = mdates.MonthLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.xticks(rotation=0)
        for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
            tick.tick1line.set_markersize(3)
            tick.tick2line.set_markersize(3)
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_horizontalalignment('center')
        #Save
        file = pl.Path(out_dir, f'{year}_clcpa_carbon_free.png')
        plt.savefig(file, bbox_inches='tight')
            
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
        plt.xticks([]); #plt.ylim(0,100)
        plt.savefig(pl.Path(out_dir,f'{year}_carbon_free_year.png'))
        
    @staticmethod
    def fig_plotly_carbon_free(year='2019', f='D'):
        """
        Figure Inspiration: NYISO Power Trends 2020 - Figure 12: Production of In-State Renewables & Zero-Emission Resources Relative to 2019 Load 
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
        fuel_mix['Net Imports'] = imports #Add Imports to fuel mix
    
        #Calculating Carbon-free Fraction [%]
        ef = fuel_mix.div(load, axis='index') * 100
        carbonfree_sources = ['Nuclear','Hydro','Other Renewables','Wind']
        ef['percent_carbon_free'] = ef[carbonfree_sources].sum(axis='columns')
                    
        #Plot Carbon-free Fraction
        df = ef[carbonfree_sources] #plot only carbon free resources   
        df = df.stack().reset_index().rename(columns={0:'Value'})
        fig = px.area(df, x='Time Stamp', y='Value',
                        line_group='Fuel Category', color='Fuel Category',
                        color_discrete_map={x:LEGEND_DEETS.get(x, '#333333') for x in df['Fuel Category'].unique()},
                        title="Daily Carbon-Free Energy",
                        labels={'Time Stamp': '', 'Value': 'Percent of Load Served by Carbon-Free Energy'},
                        width=1000, height=500)
        fig.update_yaxes(range=[0, 100])

        plot_div = opy.plot(fig,
                            output_type='div', include_plotlyjs=False,
                            show_link=False, link_text="")
                        
        return plot_div
    
    @staticmethod
    def fig_carbon_new_free_sunburst():
        year=2019
        f='H'
        #Power [MW]
        load = NYISOData(dataset='load_5m',year=year).df.tz_convert('US/Eastern')['NYCA']
        fuel_mix = NYISOData(dataset='fuel_mix_5m',year=year).df.tz_convert('US/Eastern')
        imports = NYISOData(dataset='interface_flows_5m', year=year).df.tz_convert('US/Eastern')
        imports = imports[imports['Interface Name'].isin(EXTERNAL_TFLOWS_MAP.values())]['Flow (MW)']
    
        #Energy Converstion [MWh] and Resampling By Summing Energy
        #load = (load * 1/12).sum(axis='index')  
        fuel_mix = (fuel_mix * 1/12).sum(axis='index')   
        imports = (imports * 1/12 ).sum(axis='index')
        
        fuel_mix['Net Imports'] = imports #Add Imports to fuel mix
        #fuel_mix['Load'] = load
        
        df= fuel_mix.to_frame()
    
        # #Calculating Carbon-free Fraction [%]
        # ef = fuel_mix.div(load, axis='index') * 100
        # carbonfree_sources = ['Nuclear','Hydro','Other Renewables','Wind']
        # ef['percent_carbon_free'] = ef[carbonfree_sources].sum(axis='columns')
                    
        #Plot Carbon-free Fraction
        # df = ef[carbonfree_sources] #plot only carbon free resources   
        df = df.stack().reset_index().rename(columns={0:'Value'})
        
        df['Renewable Status'] = df['Fuel Category'].map({'Dual Fuel':'Unrenewable',
                                                          'Other Fossil Fuels':'Unrenewable',
                                                          'Natural Gas':'Unrenewable',
                                                          'Net Imports':'Unknown',
                                                          'Hydro':'Renewable',
                                                          'Wind':'Renewable',
                                                          'Other Renewables': 'Renewable',
                                                          'Nuclear':'Unrenewable'})
        
        df['Carbon Status'] = df['Fuel Category'].map({'Dual Fuel':'Carbon-Emitting',
                                                       'Other Fossil Fuels':'Carbon-Emitting',
                                                       'Natural Gas':'Carbon-Emitting',
                                                       'Net Imports':'Unknown',
                                                       'Hydro':'Carbon-Free',
                                                       'Wind':'Carbon-Free',
                                                       'Other Renewables': 'Carbon-Free',
                                                       'Nuclear':'Carbon-Free'})
        
        # df2 = df.groupby('Carbon Status').sum()
        colors = {x:LEGEND_DEETS.get(x, '#333333') for x in df['Fuel Category'].unique()}
        df['color'] = df['Fuel Category'].map(colors)
        # fig = make_subplots(1, 1, specs=[[{"type": "domain"}]])
        
        # df_all_trees = df.rename(columns={'Fuel Category':'id','Renewable Status':'parent',
        #                                   'Value':'value'})[['id','parent','value','color']]
        # df_all_trees['value'] = df_all_trees['value']/(10**6)
        # print(df_all_trees)
        # fig = go.Figure()
        
        # fig.add_trace(go.Sunburst(
        #     labels=df_all_trees['id'],
        #     parents=df_all_trees['parent'],
        #     values=df_all_trees['value'],
        #     branchvalues='total',
        #     ))
        
        
        # # fig.add_trace(go.Sunburst(
        # #     labels=df_all_trees['id'],
        # #     parents=df_all_trees['parent'],
        # #     values=df_all_trees['value'],
        # #     branchvalues='total',
        # #     marker=dict(
        # #         colors=df_all_trees['color'],
        # #         colorscale='RdBu',
        # #         cmid=average_score),
        # #     hovertemplate='<b>%{label} </b> <br> Sales: %{value}<br> Success rate: %{color:.2f}',
        # #     name=''
        # #     ))
        
        # # fig = px.sunburst(
        # #                   labels=df['Fuel Category'],
        # #                   names=df['Fuel Category'],
        # #                   parents=df['Renewable Status'],
        # #                   values=df['Value'],
        # #                   branchvalues='total',
        # #                   color=df['color'],
        # #                   width=1000, height=500,
        # #                   title='Total Decarb'
        # #                   # marker=dict(
        # #                   # color_discrete_map=colors
        # #                         # colorscale='RdBu',
        # #                         # # cmid=df2,
        # #                         # ),
        # #                     # hovertemplate='<b>%{label} </b> <br> Sales: %{value}<br> Success rate: %{color:.2f}',
        # #                   )
                    
        
        
        colors = {x:LEGEND_DEETS.get(x, '#333333') for x in df['Fuel Category'].unique()}
        colors.update({'Carbon-Emitting':'#333333',
                        'Renewable':'limegreen',
                        'Carbon-Free':'lawngreen'})
        fig = px.sunburst(df, path=['Fuel Category', 'Renewable Status', 'Carbon Status'],
                          values='Value', color='Fuel Category', color_discrete_map=colors)
        
        
        
        # df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/sales_success.csv')
        # print(df.head())
        
        # levels = ['salesperson', 'county', 'region'] # levels used for the hierarchical chart
        # color_columns = ['sales', 'calls']
        # value_column = 'calls'
        
        # def build_hierarchical_dataframe(df, levels, value_column, color_columns=None):
        #     """
        #     Build a hierarchy of levels for Sunburst or Treemap charts.
        
        #     Levels are given starting from the bottom to the top of the hierarchy,
        #     ie the last level corresponds to the root.
        #     """
        #     df_all_trees = pd.DataFrame(columns=['id', 'parent', 'value', 'color'])
        #     for i, level in enumerate(levels):
        #         df_tree = pd.DataFrame(columns=['id', 'parent', 'value', 'color'])
        #         dfg = df.groupby(levels[i:]).sum()
        #         dfg = dfg.reset_index()
        #         df_tree['id'] = dfg[level].copy()
        #         if i < len(levels) - 1:
        #             df_tree['parent'] = dfg[levels[i+1]].copy()
        #         else:
        #             df_tree['parent'] = 'total'
        #         df_tree['value'] = dfg[value_column]
        #         df_tree['color'] = dfg[color_columns[0]] / dfg[color_columns[1]]
        #         df_all_trees = df_all_trees.append(df_tree, ignore_index=True)
        #     total = pd.Series(dict(id='total', parent='',
        #                               value=df[value_column].sum(),
        #                               color=df[color_columns[0]].sum() / df[color_columns[1]].sum()))
        #     df_all_trees = df_all_trees.append(total, ignore_index=True)
        #     return df_all_trees
        
        
        # df_all_trees = build_hierarchical_dataframe(df, levels, value_column, color_columns)
        # print(df_all_trees.head(30))
        # average_score = df['sales'].sum() / df['calls'].sum()
        
        # fig = go.Figure()
        
        # fig.add_trace(go.Sunburst(
        #     labels=df_all_trees['id'],
        #     parents=df_all_trees['parent'],
        #     values=df_all_trees['value'],
        #     branchvalues='total',
        #     ))
        

        plot_div = opy.plot(fig, output_type='div', include_plotlyjs=False,
                            show_link=False, link_text="")
        return plot_div
    
    
    @staticmethod
    def fig_carbon_free_sunburst():
        year=2019
        f='H'
        #Power [MW]
        load = NYISOData(dataset='load_5m',year=year).df.tz_convert('US/Eastern')['NYCA']
        fuel_mix = NYISOData(dataset='fuel_mix_5m',year=year).df.tz_convert('US/Eastern')
        # imports = NYISOData(dataset='interface_flows_5m', year=year).df.tz_convert('US/Eastern')
        # imports = imports[imports['Interface Name'].isin(EXTERNAL_TFLOWS_MAP.values())]['Flow (MW)']
    
        #Energy Converstion [MWh] and Resampling By Summing Energy
        load = (load * 1/12).sum(axis='index')  
        fuel_mix = (fuel_mix * 1/12).sum(axis='index')   
        # imports = (imports * 1/12 ).sum(axis='index')
        
        fuel_mix['Net Imports'] = (load-fuel_mix.sum()) #Add Imports to fuel mix, and clip net imports
        #fuel_mix['Load'] = load
        
        df= fuel_mix.to_frame()
        
        # #Calculating Carbon-free Fraction [%]
        # ef = fuel_mix.div(load, axis='index') * 100
        # carbonfree_sources = ['Nuclear','Hydro','Other Renewables','Wind']
        # ef['percent_carbon_free'] = ef[carbonfree_sources].sum(axis='columns')
                    
        #Plot Carbon-free Fraction
        # df = ef[carbonfree_sources] #plot only carbon free resources   
        df = df.stack().reset_index().rename(columns={0:'Value'})
        
        df['Renewable Status'] = df['Fuel Category'].map({'Dual Fuel':'Unrenewable',
                                                          'Other Fossil Fuels':'Unrenewable',
                                                          'Natural Gas':'Unrenewable',
                                                          'Net Imports':'Unknown',
                                                          'Hydro':'Renewable',
                                                          'Wind':'Renewable',
                                                          'Other Renewables': 'Renewable',
                                                          'Nuclear':'Unrenewable'})
        
        df['Carbon Status'] = df['Fuel Category'].map({'Dual Fuel':'Carbon-Emitting',
                                                       'Other Fossil Fuels':'Carbon-Emitting',
                                                       'Natural Gas':'Carbon-Emitting',
                                                       'Net Imports':'Unknown',
                                                       'Hydro':'Carbon-Free',
                                                       'Wind':'Carbon-Free',
                                                       'Other Renewables': 'Carbon-Free',
                                                       'Nuclear':'Carbon-Free'})
        
        df['Total'] = load
        colors = {x:LEGEND_DEETS.get(x, '#333333') for x in df['Fuel Category'].unique()}
        colors.update({'Carbon-Emitting':'#333333',
                       'Renewable':'limegreen',
                       'Carbon-Free':'lawngreen'})
        fig = px.sunburst(df, path=['Total', 'Carbon Status', 'Renewable Status', 'Fuel Category'],
                          values='Value', branchvalues='total')
        plot_div = opy.plot(fig, output_type='div', include_plotlyjs=False,
                            show_link=False, link_text="")
        return plot_div
    
    def fig_carbon_free_years(years):
        """Todo: stacked area chart over time using nyisostat annual summary"""
        return
    
    def net_load():
        """Todo: Load vs Net Load Shape"""
        return
            
if __name__ == '__main__':
    # NYISOVis.fig_energy(year='2019', f='D')
    #NYISOVis.fig_carbon_free_year(year='2019')
    NYISOVis.fig_clcpa_carbon_free(year='2019', f='D')
    # NYISOVis.fig_plotly_clcpa_carbon_free()
    NYISOVis.fig_carbon_free_sunburst()
    
