# NYISOData
Tool for accessing power system data from the New York Independent System Operator.

Datasets Currently Supported:
- load_h  (hourly load by NYISO region)
- load_5m (5-min load by NYISO region)
- fuel_mix_5m (5-min frequency)
- interface_flows_5m (5-min internal and external flows between regions)

Units for all datasets...
- Values: Power [MW]
- Timezone: Coordinated Universal Time [UTC]

# Usage Example
```python
from nyisodata import NYISOData
df = NYISOData(dataset='load_h', year='2019').df # year argument in local time, but returns dataset in UTC 

#if you need to work in locat time, then convert time zone
df = df.tz_localize(US/Eastern)
```

# Raw Data Information

Real-Time Hourly Actual Load
- "Integrated Real-Time Actual Load is posted after each hour and represents the timeweighted hourly load for each zone" - NYISO Manual
- Frequency: Hourly (Sometimes they may miss or do higher)
- Datetime Convention: Start of hour

Real-Time 5-min Actual Load
- "Real-Time Actual Load posts the actual measured load for each RTD interval (5 minutes) by zone. 
Actual loads are calculated as generation plus net interchange for each zone, based on real-time telemetered data." - NYISO Manual
- Frequency: 5 mins (Sometimes they may miss or do higher)
- Datetime Convention: End of 5 mins (Thats what I interpreted from the timing of release of realtime data)

Energy Mix
- Datetime convention: End of 5 mins

Interface flows
- Positive and Negative limits are currently not being pulled
- Datetime Convention: End of 5 mins
