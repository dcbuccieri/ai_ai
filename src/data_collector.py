import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import pytz
from .feature_engineer import enhance_price_data

def get_data_for_date(ticker, date, timeframe):
    """Get data for a specific ticker and date, downloading only if not already stored locally."""
    # Skip weekends (Saturday=5, Sunday=6)
    if date.weekday() >= 5:
        print(f"Skipping {date.strftime('%Y-%m-%d')} (weekend)")
        return None
    
    data_dir = f"data/{ticker}/{timeframe}"
    os.makedirs(data_dir, exist_ok=True)
    
    filename = f"{data_dir}/{date.strftime('%Y-%m-%d')}.pkl"
    
    # Check if file already exists
    if os.path.exists(filename):
        print(f"Loading {ticker} data for {date.strftime('%Y-%m-%d')} from local file")
        data = pd.read_pickle(filename)
        return data
    else:
        print(f"Downloading {ticker} data for {date.strftime('%Y-%m-%d')} from yfinance")
        try:
            # Download data for the specific date
            data = yf.download(ticker, start=date, end=date + timedelta(days=1), 
                            interval=timeframe, auto_adjust=False)
            
            if not data.empty:
                # Filter data to only include the specific date
                target_date = date.date()
                filtered_data = data[data.index.date == target_date]
                
                if not filtered_data.empty:
                    # Convert timestamps to EST
                    est = pytz.timezone('US/Eastern')
                    if filtered_data.index.tz is None:
                        filtered_data.index = filtered_data.index.tz_localize('UTC').tz_convert(est)
                    else:
                        filtered_data.index = filtered_data.index.tz_convert(est)
                    
                    # Save only the filtered data to local file
                    filtered_data.to_pickle(filename)
                    print(f"  Saved {len(filtered_data)} records for {date.strftime('%Y-%m-%d')} (EST)")
                    return filtered_data
                else:
                    print(f"  No data found for {date.strftime('%Y-%m-%d')}")
                    return None
            else:
                print(f"  No data available for {ticker} on {date.strftime('%Y-%m-%d')}")
                return None
        except Exception as e:
            print(f"  Error: {e}")
            return None

def collect_data_for_date_range(ticker, timeframe, start_date, end_date):
    """Collect data for a ticker over a specified date range."""
    print(f"Checking data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    current_date = start_date
    total_days = 0
    
    while current_date < end_date:
        data = get_data_for_date(ticker, current_date, timeframe)
        if data is not None and not data.empty:
            total_days += 1
        current_date += timedelta(days=1)
    
    print(f"Total trading days available: {total_days}")
    return total_days

def load_first_and_last_data(ticker, timeframe):
    """Load and display the first and last available data files with enhanced features."""
    data_dir = f"data/{ticker}/{timeframe}"
    files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    
    if files:
        first_file = sorted(files)[0]  # Get the first (earliest) file
        last_file = sorted(files)[-1]  # Get the last (latest) file
        
        # Extract dates from filenames
        first_date = datetime.strptime(first_file.replace('.pkl', ''), '%Y-%m-%d')
        last_date = datetime.strptime(last_file.replace('.pkl', ''), '%Y-%m-%d')
        
        # Load and enhance first file
        first_data = pd.read_pickle(f"{data_dir}/{first_file}")
        first_enhanced = enhance_price_data(first_data, ticker, first_date, timeframe)
        
        print(f"\nFirst 5 rows from {first_file} (EST) with features:")
        print(first_enhanced.head())
        
        # Load and enhance last file
        last_data = pd.read_pickle(f"{data_dir}/{last_file}")
        last_enhanced = enhance_price_data(last_data, ticker, last_date, timeframe)
        
        print(f"\nLast 5 rows from {last_file} (EST) with features:")
        print(last_enhanced.tail())
        
        return first_enhanced, last_enhanced
    else:
        print("No data files found")
        return None, None
