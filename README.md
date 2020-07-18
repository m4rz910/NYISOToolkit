# NYISOData
Tool for accessing power system data from the New York Independent System Operator.

Datasets Currently Supported:
- load_h  (hourly load by NYISO region)
- load_5m (5-min load by NYISO region)
- fuel_mix_5m (5-min frequency)
- interface_flows_5m (5-min internal and external flows between regions)

All datasets...
- Values: Power [MW]
- Timezone: Coordinated Universal Time [UTC]
- Frequency: Raw data sometimes has higher or lower frequency than it states but the code resamples using mean values

# Usage Example
```python
from nyisodata import NYISOData
df = NYISOData(dataset='load_h', year='2019').df # year argument in local time, but returns dataset in UTC 

#if you need to work in local time, then convert time zone
df = df.tz_localize('US/Eastern')
```

# Dataset Information

Real-Time Hourly Actual Load (load_h)
- "Integrated Real-Time Actual Load is posted after each hour and represents the timeweighted hourly load for each zone" - NYISO Manual
- Frequency: Hourly
- Datetime Convention: Start of hour

Real-Time 5-min Actual Load (load_5m)
- "Real-Time Actual Load posts the actual measured load for each RTD interval (5 minutes) by zone. 
Actual loads are calculated as generation plus net interchange for each zone, based on real-time telemetered data." - NYISO Manual
- Frequency: 5 mins 
- Datetime Convention: End of 5 mins (That's what I interpreted from the timing of release of realtime data)

Fuel Mix (fuel_mix_5m)
- Frequency: 5 mins 
- Datetime convention: End of 5 mins

Interface flows (interface_flows_5m)
- Positive and Negative limits are currently not included
- Frequency: 5 mins 
- Datetime Convention: End of 5 mins
