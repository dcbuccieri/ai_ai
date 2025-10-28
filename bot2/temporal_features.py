"""
Temporal Features - Time-based features for stock prediction

Captures cyclical patterns in trading (time of day, day of week, etc.)
"""
import pandas as pd
import numpy as np
from datetime import datetime
from config import MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE, MARKET_CLOSE_HOUR
from utils import setup_logging

logger = setup_logging(__name__)

def add_time_of_day_features(df):
    """
    Add time of day features using sin/cos encoding for cyclical nature
    
    Args:
        df: DataFrame with datetime index
    
    Returns:
        DataFrame: Original data with time features added
    """
    df = df.copy()
    
    # Extract hour and minute
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    
    # Minutes since market open (9:30 AM = 0)
    market_open_minutes = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE
    df['minutes_since_open'] = df['hour'] * 60 + df['minute'] - market_open_minutes
    
    # Cyclical encoding (sin/cos) for hour (24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for minute (60-minute cycle)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    
    # Market session (categorical)
    df['market_session'] = 'mid_day'
    df.loc[df['minutes_since_open'] < 30, 'market_session'] = 'opening'
    df.loc[df['minutes_since_open'] < 60, 'market_session'] = 'early'
    df.loc[df['minutes_since_open'] > 330, 'market_session'] = 'closing'
    
    # One-hot encode market session
    session_dummies = pd.get_dummies(df['market_session'], prefix='session')
    df = pd.concat([df, session_dummies], axis=1)
    df.drop('market_session', axis=1, inplace=True)
    
    # Drop raw hour/minute (we have encoded versions)
    df.drop(['hour', 'minute'], axis=1, inplace=True)
    
    return df

def add_day_of_week_features(df):
    """
    Add day of week features
    
    Args:
        df: DataFrame with datetime index
    
    Returns:
        DataFrame: Original data with day features added
    """
    df = df.copy()
    
    # Day of week (0=Monday, 4=Friday)
    df['day_of_week'] = df.index.dayofweek
    
    # Cyclical encoding
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
    
    # Binary flags for Monday and Friday (different trading patterns)
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    # Drop raw day_of_week
    df.drop('day_of_week', axis=1, inplace=True)
    
    return df

def add_earnings_features(df, ticker, earnings_dates=None):
    """
    Add features related to earnings announcements
    
    Args:
        df: DataFrame with datetime index
        ticker: Stock ticker symbol
        earnings_dates: Optional dict mapping dates to earnings info
    
    Returns:
        DataFrame: Original data with earnings features added
    """
    df = df.copy()
    
    # For now, use a simple quarterly estimate (90 days)
    # In production, integrate with earnings calendar API
    
    # Get the date from index
    if len(df) > 0:
        current_date = df.index[0].date()
        
        # Simple approximation: Companies report quarterly
        # Assume earnings dates are roughly every 90 days
        # For now, use a fixed offset (can be improved with actual calendar)
        days_in_quarter = 90
        days_into_quarter = (current_date.toordinal() % days_in_quarter)
        days_until_earnings = days_in_quarter - days_into_quarter
        
        df['days_until_earnings'] = days_until_earnings
        df['earnings_soon'] = int(days_until_earnings < 7)  # Within a week
    else:
        df['days_until_earnings'] = 45  # Mid-quarter default
        df['earnings_soon'] = 0
    
    return df

def add_month_features(df):
    """
    Add month-related features (some months have different patterns)
    
    Args:
        df: DataFrame with datetime index
    
    Returns:
        DataFrame: Original data with month features added
    """
    df = df.copy()
    
    df['month'] = df.index.month
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # January effect (historically different pattern)
    df['is_january'] = (df['month'] == 1).astype(int)
    
    # Quarter
    df['quarter'] = df.index.quarter
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    
    df.drop('month', axis=1, inplace=True)
    
    return df

def add_market_regime_features(df):
    """
    Add features to detect market regime (trend direction and strength)
    
    Args:
        df: DataFrame with Close prices
    
    Returns:
        DataFrame: Original data with regime features added
    """
    df = df.copy()
    
    # Short, medium, and long term trends
    df['trend_short'] = df['Close'].pct_change(periods=5)  # 5-minute trend
    df['trend_medium'] = df['Close'].pct_change(periods=30)  # 30-minute trend
    df['trend_long'] = df['Close'].pct_change(periods=60)  # 60-minute trend
    
    # Volatility regime (rolling std of returns)
    returns = df['Close'].pct_change()
    df['volatility_regime'] = returns.rolling(window=30).std()
    
    # High volatility flag
    df['high_volatility'] = (df['volatility_regime'] > df['volatility_regime'].rolling(window=100).mean()).astype(int)
    
    return df

def add_all_temporal_features(df, ticker=None):
    """
    Add all temporal features to a DataFrame
    
    Args:
        df: DataFrame with OHLCV data and datetime index
        ticker: Optional ticker symbol for earnings features
    
    Returns:
        DataFrame: Original data with all temporal features added
    """
    logger.info("Adding temporal features...")
    
    df = add_time_of_day_features(df)
    df = add_day_of_week_features(df)
    df = add_month_features(df)
    df = add_market_regime_features(df)
    
    if ticker:
        df = add_earnings_features(df, ticker)
    
    logger.info(f"Added temporal features, total columns: {len(df.columns)}")
    
    return df

if __name__ == '__main__':
    # Test with sample data
    import pandas as pd
    import numpy as np
    
    # Create sample data spanning multiple days
    dates = pd.date_range('2024-01-02 09:30', periods=390, freq='1min')  # One trading day
    
    df = pd.DataFrame({
        'Open': 100 + np.random.randn(390).cumsum() * 0.1,
        'High': 101 + np.random.randn(390).cumsum() * 0.1,
        'Low': 99 + np.random.randn(390).cumsum() * 0.1,
        'Close': 100 + np.random.randn(390).cumsum() * 0.1,
        'Volume': np.random.randint(1000, 10000, 390)
    }, index=dates)
    
    df_with_temporal = add_all_temporal_features(df, ticker='AAPL')
    
    print("\nSample data with temporal features:")
    print(df_with_temporal.head(10))
    print(f"\nTotal columns: {len(df_with_temporal.columns)}")
    print(f"\nTemporal feature columns:")
    temporal_cols = [col for col in df_with_temporal.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    print(temporal_cols)

