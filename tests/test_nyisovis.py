import pytest

from nyisotoolkit import NYISOVis

def test_vis(years = ['2019']):
    for year in years:
            nv = NYISOVis(year=year)
            nv.fig_carbon_free_year()
            for f in ['D','M']:
                nv.fig_energy(f=f)
                nv.fig_clcpa_carbon_free(f=f)