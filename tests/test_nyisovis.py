import pytest
from datetime import datetime
import pytz
import matplotlib.pyplot as plt

from nyisotoolkit import NYISOVis
from nyisotoolkit.nyisovis.nyisovis import basic_plots, statistical_plots
from nyisotoolkit.nyisodata.utils import current_year

def test_basic_plots():
    nyisovis_kwargs = {"redownload":True}
    years=list(range(2021, current_year()+1))
    for year in years:
        nyisovis_kwargs["year"] = year
        basic_plots(nyisovis_kwargs)
        plt.close("all")
            
def test_statistical_plots():
    nyisovis_kwargs = {"redownload":False}
    years=list(range(2021, current_year()+1))
    for year in years:
        nyisovis_kwargs["year"] = year
        statistical_plots(nyisovis_kwargs)
        plt.close("all")
    
if __name__ == "__main__":
    test_basic_plots()
    test_statistical_plots()