from src.data_collector import collect_data_for_date_range, load_first_and_last_data
from src.feature_engineer import enhance_price_data
from datetime import datetime

# Configuration variables - modify these to test different inputs
TICKER = 'AAPL'
TIMEFRAME = '1m'  # Options: '1m', '5m', '15m', '1h', '1d'
START_DATE = datetime(2025, 9, 27)
END_DATE = datetime(2025, 10, 25)  # Go to 25th to include 24th

# Collect data using the data collector module
total_days = collect_data_for_date_range(TICKER, TIMEFRAME, START_DATE, END_DATE)

# Load and display the first and last available data
if total_days > 0:
    load_first_and_last_data(TICKER, TIMEFRAME)
