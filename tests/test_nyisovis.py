import pytest

from nyisotoolkit import NYISOVis

def test_vis(years = ['2020']):
    for year in years:
            nv = NYISOVis(year=year)
            nv.fig_carbon_free_year()
            nv.fig_decarbonization_heatmap()
            for f in ['D','M']:
                nv.fig_energy_generation(f=f)
                nv.fig_carbon_free_timeseries(f=f)