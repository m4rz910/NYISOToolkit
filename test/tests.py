# -*- coding: utf-8 -*-

from nyisodata import NYISOData

if __name__ == '__main__':
    #df = NYISOData(dataset='interface_flows_5m', year='2013', reconstruct=True).df
    df = NYISOData(dataset='lbmp_dam_h', year='2019', reconstruct=True).df
    df = NYISOData(dataset='lbmp_rt_5m', year='2019', reconstruct=True).df