# Market Hours Filtering

## Overview

The system downloads **all available data** (pre-market, regular hours, and after-hours) from Polygon.io, but **only uses regular market hours** (9:30 AM - 4:00 PM ET) for training and prediction.

## Why Download Extended Hours?

- **Data preservation**: You have the data if you want to use it later
- **Future flexibility**: Can analyze pre/post-market patterns if needed
- **No extra cost**: Same API call gets all the data

## Why Train on Regular Hours Only?

- **Consistent patterns**: Regular hours have more volume and liquidity
- **Reduced noise**: Pre/post-market can be very volatile and thin
- **Standard trading**: Most retail traders operate during regular hours
- **Better generalization**: Model learns stable patterns

## How It Works

### 1. Data Download (`data_downloader.py`)
Downloads everything: ~960 minutes per day (4 AM - 8 PM)

```bash
python data_downloader.py --ticker AAPL --days 30
# Saves ~645 rows per day (includes extended hours)
```

### 2. Feature Processing (`feature_pipeline.py`)
Automatically filters to regular hours: 390 minutes per day

```bash
python feature_pipeline.py --ticker AAPL --start-date ... --end-date ...
# Processes only 9:30 AM - 4:00 PM data
# Logs: "Filtered out 255 pre/after-market rows, keeping 390 regular hours"
```

### 3. Live Prediction (`live_predictor.py`)
Also filters to regular hours automatically

```bash
python live_predictor.py --ticker AAPL
# Only makes predictions during 9:30 AM - 4:00 PM
```

## Expected Row Counts

### Raw Downloaded Data (per day)
- **Full extended hours**: ~960 rows
- **Typical actual data**: 600-800 rows (depends on market activity)
  - Pre-market (4:00-9:30 AM): ~150-250 rows
  - Regular hours (9:30 AM-4:00 PM): 390 rows
  - After-hours (4:00-8:00 PM): ~100-200 rows

### Processed Training Data (per day)
- **After filtering**: ~390 rows (regular hours only)
- **After NaN cleanup**: ~360-380 rows (some lost to indicator calculations)

## Verification

To verify the filter is working:

```python
from utils import filter_regular_hours
import pandas as pd

# Load raw data
df = pd.read_pickle('data/AAPL/raw/1m/2024-02-08.pkl')
print(f"Raw data: {len(df)} rows")
print(f"Time range: {df.index.min()} to {df.index.max()}")

# Filter to regular hours
filtered = filter_regular_hours(df)
print(f"\nFiltered data: {len(filtered)} rows")
print(f"Time range: {filtered.index.min()} to {filtered.index.max()}")
print(f"Expected: 390 rows from 9:30 AM to 3:59 PM")
```

## Configuration

Market hours are defined in `config.py`:

```python
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
```

The filter function in `utils.py` uses these settings:

```python
def filter_regular_hours(df, timezone='US/Eastern'):
    """Filter to 9:30 AM - 4:00 PM ET only"""
    mask = (df.index.hour > 9) | ((df.index.hour == 9) & (df.index.minute >= 30))
    mask &= (df.index.hour < 16)
    return df[mask].copy()
```

## Advanced: Using Extended Hours

If you want to train on extended hours data in the future, simply comment out the filter line in `feature_pipeline.py`:

```python
# df = filter_regular_hours(df)  # Comment this out to use all data
```

But for now, **regular hours only is the recommended approach** for most accurate predictions.

## Summary

✅ **Downloads**: All available data (extended hours included)  
✅ **Trains on**: Regular hours only (9:30 AM - 4:00 PM)  
✅ **Result**: ~390 rows per day used for training  
✅ **Benefit**: Best of both worlds - data preservation + clean training  

Your model will learn from the most liquid, consistent trading periods while preserving the full dataset for future use.

