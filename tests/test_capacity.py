import pandas as pd
from nyisotoolkit.nyisodata.capacity import NYISOCapacity

def test_download(dates = ['2023-11-01',
                           '2024-08-01']):
    for date in dates:
        NYISOCapacity(date=date).get_raw_data()
        
def test_dataframes():
    o=NYISOCapacity(date=pd.Timestamp.now())
    o.prices()
    o.summary_table()
    o.ucap_table()

if __name__ == '__main__':
    o = NYISOCapacity(date=pd.Timestamp.now())
    #o.test_download()
    df = o.ucap_table()
    df