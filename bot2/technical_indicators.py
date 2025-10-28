"""
Technical Indicators - Calculate technical analysis features

Implements common TA indicators without external TA-Lib dependency.
All indicators are vectorized using pandas/numpy for performance.
"""
import pandas as pd
import numpy as np
from utils import setup_logging

logger = setup_logging(__name__)

def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index
    
    Args:
        prices: Series of closing prices
        period: RSI period (default 14)
    
    Returns:
        Series: RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        prices: Series of closing prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
    
    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """
    Calculate Bollinger Bands
    
    Args:
        prices: Series of closing prices
        period: Moving average period
        std_dev: Number of standard deviations
    
    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band

def calculate_atr(high, low, close, period=14):
    """
    Calculate Average True Range
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: ATR period
    
    Returns:
        Series: ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: ADX period
    
    Returns:
        Series: ADX values
    """
    # Calculate +DM and -DM
    high_diff = high.diff()
    low_diff = -low.diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_stochastic(high, low, close, period=14, smooth_k=3):
    """
    Calculate Stochastic Oscillator
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: Lookback period
        smooth_k: Smoothing for %K
    
    Returns:
        tuple: (%K, %D)
    """
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    k = k.rolling(window=smooth_k).mean()
    d = k.rolling(window=3).mean()
    
    return k, d

def calculate_obv(close, volume):
    """
    Calculate On-Balance Volume
    
    Args:
        close: Series of closing prices
        volume: Series of volume
    
    Returns:
        Series: OBV values
    """
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def calculate_vwap(high, low, close, volume):
    """
    Calculate Volume Weighted Average Price
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        volume: Series of volume
    
    Returns:
        Series: VWAP values
    """
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

def calculate_ema(prices, period):
    """
    Calculate Exponential Moving Average
    
    Args:
        prices: Series of prices
        period: EMA period
    
    Returns:
        Series: EMA values
    """
    return prices.ewm(span=period, adjust=False).mean()

def calculate_sma(prices, period):
    """
    Calculate Simple Moving Average
    
    Args:
        prices: Series of prices
        period: SMA period
    
    Returns:
        Series: SMA values
    """
    return prices.rolling(window=period).mean()

def calculate_momentum(prices, period=10):
    """
    Calculate Momentum (rate of change)
    
    Args:
        prices: Series of prices
        period: Lookback period
    
    Returns:
        Series: Momentum values
    """
    return prices.diff(period)

def calculate_roc(prices, period=10):
    """
    Calculate Rate of Change (percentage)
    
    Args:
        prices: Series of prices
        period: Lookback period
    
    Returns:
        Series: ROC values (percentage)
    """
    return ((prices - prices.shift(period)) / prices.shift(period)) * 100

def add_all_indicators(df):
    """
    Add all technical indicators to a DataFrame
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame: Original data with all indicators added
    """
    logger.info("Calculating technical indicators...")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # RSI
    df['rsi_14'] = calculate_rsi(df['Close'], 14)
    df['rsi_28'] = calculate_rsi(df['Close'], 28)
    
    # MACD
    macd, signal, hist = calculate_macd(df['Close'])
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = hist
    
    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(df['Close'])
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_width'] = (upper - lower) / middle  # Normalized width
    
    # ATR
    df['atr'] = calculate_atr(df['High'], df['Low'], df['Close'])
    df['atr_pct'] = df['atr'] / df['Close']  # ATR as percentage of price
    
    # ADX
    df['adx'] = calculate_adx(df['High'], df['Low'], df['Close'])
    
    # Stochastic
    k, d = calculate_stochastic(df['High'], df['Low'], df['Close'])
    df['stoch_k'] = k
    df['stoch_d'] = d
    
    # Volume indicators
    df['obv'] = calculate_obv(df['Close'], df['Volume'])
    df['vwap'] = calculate_vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    df['volume_sma'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = calculate_sma(df['Close'], period)
        df[f'ema_{period}'] = calculate_ema(df['Close'], period)
    
    # Price position relative to moving averages
    df['price_above_sma20'] = (df['Close'] > df['sma_20']).astype(int)
    df['price_above_sma50'] = (df['Close'] > df['sma_50']).astype(int)
    
    # Momentum
    df['momentum'] = calculate_momentum(df['Close'], 10)
    df['roc'] = calculate_roc(df['Close'], 10)
    
    # Price changes
    df['close_change'] = df['Close'].pct_change()
    df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
    
    logger.info(f"Added {len(df.columns) - 6} technical indicators")  # -6 for OHLCV + index
    
    return df

if __name__ == '__main__':
    # Test with sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'Open': 100 + np.random.randn(100).cumsum(),
        'High': 101 + np.random.randn(100).cumsum(),
        'Low': 99 + np.random.randn(100).cumsum(),
        'Close': 100 + np.random.randn(100).cumsum(),
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    df_with_indicators = add_all_indicators(df)
    
    print("\nSample data with indicators:")
    print(df_with_indicators.head())
    print(f"\nTotal columns: {len(df_with_indicators.columns)}")
    print(f"Column names: {df_with_indicators.columns.tolist()}")

