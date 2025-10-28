import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np

def add_earnings_feature(price_data, ticker, date):
    """
    Add 'days_until_earnings' feature to price data.
    
    Args:
        price_data: DataFrame with price data
        ticker: Stock ticker symbol
        date: Date for which to calculate earnings feature
    
    Returns:
        DataFrame with additional 'days_until_earnings' column
    """
    # Get earnings calendar for the ticker
    try:
        stock = yf.Ticker(ticker)
        earnings_dates = stock.calendar
        
        if earnings_dates is not None:
            # yfinance returns a dict with 'Earnings Date' key
            if isinstance(earnings_dates, dict) and 'Earnings Date' in earnings_dates:
                earnings_list = earnings_dates['Earnings Date']
                if earnings_list and len(earnings_list) > 0:
                    # Get the next earnings date after the given date
                    next_earnings = None
                    for earnings_date in earnings_list:
                        if hasattr(earnings_date, 'date'):
                            earnings_date_obj = earnings_date.date()
                        else:
                            earnings_date_obj = earnings_date
                        
                        if earnings_date_obj > date.date():
                            next_earnings = earnings_date_obj
                            break
                    
                    if next_earnings:
                        days_until = (next_earnings - date.date()).days
                    else:
                        # If no future earnings found, use a default value
                        days_until = 90  # Default to 90 days
                else:
                    days_until = 90
            else:
                days_until = 90
        else:
            # If no earnings data available, use default
            days_until = 90
            
    except Exception as e:
        print(f"Warning: Could not get earnings data for {ticker}: {e}")
        days_until = 90  # Default fallback
    
    # Add the feature to all rows in the price data
    enhanced_data = price_data.copy()
    enhanced_data['days_until_earnings'] = days_until
    
    return enhanced_data

def add_rsi_feature(price_data, timeframe):
    """
    Add RSI feature to price data.
    
    Args:
        price_data: DataFrame with price data
        timeframe: Timeframe string ('1m', '5m', '15m', '1h', '1d')
    
    Returns:
        DataFrame with additional 'rsi' column
    """
    enhanced_data = price_data.copy()
    
    # Calculate RSI using closing prices
    rsi_values = calculate_rsi(enhanced_data['Close'])
    enhanced_data['rsi'] = rsi_values
    
    return enhanced_data

def calculate_rsi(prices, period=14):
    """
    Calculate RSI (Relative Strength Index) for a price series.
    
    Args:
        prices: Series of closing prices
        period: RSI period (default 14)
    
    Returns:
        Series of RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def add_period_change_features(price_data, timeframe):
    """
    Add forward-looking period change features to price data.
    
    Args:
        price_data: DataFrame with price data
        timeframe: Timeframe string ('1m', '5m', '15m', '1h', '1d')
    
    Returns:
        DataFrame with additional period change columns
    """
    enhanced_data = price_data.copy()
    
    # Get the adjusted close price column
    close_col = 'Adj Close' if 'Adj Close' in enhanced_data.columns else 'Close'
    close_prices = enhanced_data[close_col]
    
    # Calculate period changes for different look-ahead periods
    periods = [1, 3, 5, 10]
    
    for period in periods:
        col_name = f"{period}_period_change"
        
        # Calculate forward-looking percent change
        future_prices = close_prices.shift(-period)  # Look forward 'period' rows
        period_change = (future_prices - close_prices) / close_prices
        
        # Store the result (NaN values will be automatically handled)
        enhanced_data[col_name] = period_change
    
    return enhanced_data

def enhance_price_data(price_data, ticker, date, timeframe='1m'):
    """
    Main function to enhance price data with additional features.
    Currently adds: days_until_earnings, rsi, period_change_features
    
    Args:
        price_data: DataFrame with OHLCV price data
        ticker: Stock ticker symbol
        date: Date of the price data
        timeframe: Timeframe string ('1m', '5m', '15m', '1h', '1d')
    
    Returns:
        Enhanced DataFrame with additional features
    """
    enhanced_data = price_data.copy()
    
    # Add earnings feature
    enhanced_data = add_earnings_feature(enhanced_data, ticker, date)
    
    # Add RSI feature
    enhanced_data = add_rsi_feature(enhanced_data, timeframe)
    
    # Add period change features
    enhanced_data = add_period_change_features(enhanced_data, timeframe)
    
    return enhanced_data
