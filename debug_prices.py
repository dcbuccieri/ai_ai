import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_daily_prices(ticker, date):
    """Get opening and closing prices for a specific ticker and date."""
    # Skip weekends (Saturday=5, Sunday=6)
    if date.weekday() >= 5:
        print(f"Skipping {date.strftime('%Y-%m-%d')} (weekend)")
        return None
    
    print(f"Getting {ticker} data for {date.strftime('%Y-%m-%d')}")
    
    try:
        # Download data for the specific date
        data = yf.download(ticker, start=date, end=date + timedelta(days=1), 
                        interval='1m', auto_adjust=False)
        
        if not data.empty:
            # Filter data to only include the specific date
            target_date = date.date()
            filtered_data = data[data.index.date == target_date]
            
            if not filtered_data.empty:
                opening_price = filtered_data['Open'].iloc[0].item()
                closing_price = filtered_data['Close'].iloc[-1].item()
                
                print(f"  Opening: ${opening_price:.2f}")
                print(f"  Closing: ${closing_price:.2f}")
                print(f"  Records: {len(filtered_data)}")
                
                return {
                    'date': date.strftime('%Y-%m-%d'),
                    'opening': opening_price,
                    'closing': closing_price,
                    'records': len(filtered_data)
                }
            else:
                print(f"  No data found for {date.strftime('%Y-%m-%d')}")
                return None
        else:
            print(f"  No data available for {ticker} on {date.strftime('%Y-%m-%d')}")
            return None
            
    except Exception as e:
        print(f"  Error: {e}")
        return None

# Test with specific dates Oct 16-24, 2025 (starting from previous Thursday)
start_date = datetime(2025, 10, 16)
end_date = datetime(2025, 10, 25)  # Go to 25th to include 24th

print(f"Testing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print("=" * 60)

current_date = start_date
successful_days = []

while current_date < end_date:
    result = get_daily_prices('SPY', current_date)
    if result:
        successful_days.append(result)
    current_date += timedelta(days=1)

print("=" * 60)
print(f"Successfully retrieved data for {len(successful_days)} days:")
for day in successful_days:
    print(f"{day['date']}: Open=${day['opening']:.2f}, Close=${day['closing']:.2f} ({day['records']} records)")
