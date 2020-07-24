# -*- coding: utf-8 -*-

from nyisodata import NYISOData

def test_interface_flows():
    return NYISOData(dataset='interface_flows_5m', year='2013', reconstruct=True).df

if __name__ == '__main__':
    df = test_interface_flows()