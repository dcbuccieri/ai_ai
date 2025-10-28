"""
Utility functions used across the project
"""
import os
import json
import logging
import pickle
from datetime import datetime, timedelta
import pandas as pd
import pytz
from config import LOGS_DIR, LOG_FORMAT, LOG_LEVEL

def setup_logging(name, log_file=None):
    """
    Setup logging with consistent format
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional log file path
    
    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(LOGS_DIR, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(LOGS_DIR, log_file))
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)
    
    return logger

def is_market_open(dt=None, timezone='US/Eastern'):
    """
    Check if US stock market is open at given datetime
    
    Args:
        dt: datetime object (defaults to now)
        timezone: Market timezone
    
    Returns:
        bool: True if market is open
    """
    if dt is None:
        dt = datetime.now(pytz.timezone(timezone))
    elif dt.tzinfo is None:
        dt = pytz.timezone(timezone).localize(dt)
    
    # Check if weekend
    if dt.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    
    # Check market hours (9:30 AM - 4:00 PM ET)
    market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= dt < market_close

def filter_regular_hours(df, timezone='US/Eastern'):
    """
    Filter DataFrame to only include regular market hours (9:30 AM - 4:00 PM ET)
    
    Args:
        df: DataFrame with datetime index
        timezone: Market timezone
    
    Returns:
        DataFrame filtered to regular trading hours only
    """
    import pytz
    
    # Ensure timezone-aware
    if df.index.tz is None:
        tz = pytz.timezone(timezone)
        df.index = df.index.tz_localize(tz)
    
    # Filter to regular hours
    mask = (df.index.hour > 9) | ((df.index.hour == 9) & (df.index.minute >= 30))
    mask &= (df.index.hour < 16)
    
    filtered_df = df[mask].copy()
    
    return filtered_df

def get_trading_days(start_date, end_date):
    """
    Get list of trading days between start and end dates (excludes weekends)
    
    Args:
        start_date: datetime object
        end_date: datetime object
    
    Returns:
        List of datetime objects representing trading days
    """
    trading_days = []
    current = start_date
    
    while current <= end_date:
        if current.weekday() < 5:  # Monday=0, Friday=4
            trading_days.append(current)
        current += timedelta(days=1)
    
    return trading_days

def save_pickle(data, filepath):
    """
    Save data to pickle file with error handling
    
    Args:
        data: Data to save
        filepath: Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filepath):
    """
    Load data from pickle file with error handling
    
    Args:
        filepath: Path to pickle file
    
    Returns:
        Loaded data or None if file doesn't exist
    """
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_json(data, filepath):
    """
    Save data to JSON file
    
    Args:
        data: Data to save (must be JSON serializable)
        filepath: Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath):
    """
    Load data from JSON file
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Loaded data or None if file doesn't exist
    """
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)

def percentage_change(current, previous):
    """
    Calculate percentage change
    
    Args:
        current: Current value (can be Series or scalar)
        previous: Previous value (can be Series or scalar)
    
    Returns:
        Percentage change as decimal (0.05 = 5%)
    """
    import numpy as np
    # Handle Series or scalar
    result = (current - previous) / previous
    # Replace inf/nan with 0 (when previous is 0)
    if hasattr(result, 'replace'):
        result = result.replace([np.inf, -np.inf], 0).fillna(0)
    return result

def format_money(amount):
    """Format dollar amount for display"""
    return f"${amount:,.2f}"

def format_percentage(value):
    """Format percentage for display"""
    return f"{value*100:.2f}%"

class Timer:
    """Simple context manager for timing code blocks"""
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"{self.name} took {elapsed:.2f} seconds")

def validate_ticker(ticker):
    """
    Basic ticker validation
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Cleaned ticker (uppercase, stripped)
    
    Raises:
        ValueError if invalid
    """
    ticker = ticker.strip().upper()
    
    if not ticker:
        raise ValueError("Ticker cannot be empty")
    
    if not ticker.isalpha() or len(ticker) > 5:
        raise ValueError(f"Invalid ticker format: {ticker}")
    
    return ticker

def print_dataframe_info(df, name="DataFrame"):
    """Print useful info about a DataFrame"""
    print(f"\n{'='*50}")
    print(f"{name} Info")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"{'='*50}\n")

