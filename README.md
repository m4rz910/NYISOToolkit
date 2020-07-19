# NYISOData
Tool for accessing power system data from the New York Independent System Operator.
Source: http://mis.nyiso.com/public/

Datasets Currently Supported:
- load_h  (hourly load by NYISO region)
- load_5m (5-min load by NYISO region)
- fuel_mix_5m (5-min frequency)
- interface_flows_5m (5-min internal and external flows between regions)

All datasets...
- Values: Power [MW]
- Timezone: Coordinated Universal Time [UTC]
- Frequency: Hourly or 5-mins. The raw data sometimes has higher or lower frequency than intended, but this library uses mean values to resample at intended the intended frequency. When interpolations are necessary, they are made. Some datasets only come in one frequency.

# Usage Example
```python
from nyisodata import NYISOData
df = NYISOData(dataset='load_h', year='2019').df # year argument in local time, but returns dataset in UTC 

#If you need to work with data in local time, then convert time zone
df = df.tz_localize('US/Eastern')

#Construct all available datasets for certain years
years = ['2013','2019','2020']
datasets = supported_datasets # or datasets = ['interface_flows_5m']
NYISOData.construct_databases(years=years, datasets=datasets, reconstruct=True)
```

# Dataset Information

Load (load_h)
- "Integrated Real-Time Actual Load is posted after each hour and represents the timeweighted hourly load for each zone" (NYISO Market Participant Guide p.62)
- Units: Power [MW]
- Frequency: Hour
- Datetime Convention: Start of hour

Load (load_5m)
- "Real-Time Actual Load posts the actual measured load for each RTD interval (5 minutes) by zone. 
Actual loads are calculated as generation plus net interchange for each zone, based on real-time telemetered data." (NYISO Market Participant Guide p.62)
- Units: Power [MW]
- Frequency: 5-min
- Datetime Convention: End of 5 mins (That's what I interpreted from the timing of release of realtime data)

Fuel Mix (fuel_mix_5m)
- Units: Power [MW]
- Frequency: 5-min
- Datetime convention: End of 5-min

Interface Flows (interface_flows_5m)
- "Internal/ External Interface Limits and Flows consist of hourly limits (for all major internal interfaces, HQ, NE, PJM, and OH) and flows (for HQ, NE, PJM, and OH) in SCUC and time-weighted average hourly flows (for the same interfaces) in RTD. The data is posted at least day-after or sooner." (NYISO Market Participant Guide p.59)
- Units: Energy [MWh]
- Frequency: 5-min
- Datetime Convention: End of 5-min

    External Interfaces
    - HQ CHATEAUGUAY
    - HQ CEDARS
    - HQ NET
    - NPX NEW ENGLAND (NE)
    - NPX 1385 NORTHPORT (NNC)
    - NPX CROSS SOUND CABLE (CSC)
    - IESO
    - PJM KEYSTONE
    - PJM HUDSON TP
    - PJM NEPTUNE
    - PJM LINDEN VFT

    Internal Interfaces
    - CENTRAL EAST - VC
    - DYSINGER EAST
    - MOSES SOUTH
    - SPR/DUN-SOUTH
    - TOTAL EAST
    - UPNY CONED
    - WEST CENTRAL

```python
#The following map is used to mapn datafile external interface names and those on the website
external_tflows_map = {'SCH - HQ - NY': 'HQ CHATEAUGUAY',
                       'SCH - HQ_CEDARS': 'HQ CEDARS',
                       'SCH - HQ_IMPORT_EXPORT': 'HQ NET',
                       'SCH - NE - NY':  'NPX NEW ENGLAND (NE)',
                       'SCH - NPX_1385': 'NPX 1385 NORTHPORT (NNC)',
                       'SCH - NPX_CSC':  'NPX CROSS SOUND CABLE (CSC)',
                       'SCH - OH - NY':  'IESO',
                       'SCH - PJ - NY':  'PJM KEYSTONE',
                       'SCH - PJM_HTP':  'PJM HUDSON TP',
                       'SCH - PJM_NEPTUNE':'PJM NEPTUNE',
                       'SCH - PJM_VFT': 'PJM LINDEN VFT'}     
```


