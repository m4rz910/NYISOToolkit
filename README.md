# NYISOData
Tool for accessing power system data from the New York Independent System Operator.

Datasets Currently Supported:
- load_h  (hourly load by NYISO region)
- load_5m (5-min load by NYISO region)
- fuel_mix_5m (5-min frequency)
- interface_flows_5m (5-min internal and external flows between regions)

For add datasets..
- Value Units: Power [MW]
- Timezone Units: UTC

# Usage Example
```python
from nyisodata import NYISOData
df = NYISOData(dataset='load_h', year='2019').df # year argument in local time, but returns dataset in UTC 

#if you need to work in locat time, then convert time zone
df = df.tz_localize(US/Eastern)
```
