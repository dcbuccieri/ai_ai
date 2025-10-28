"""
Sentiment Features - Integrate sentiment data into feature pipeline

Combines news, social media, and economic sentiment into model features
"""
import pandas as pd
import numpy as np
from utils import setup_logging

logger = setup_logging(__name__)

def add_sentiment_features(df, sentiment_data):
    """
    Add sentiment features to price data
    
    Args:
        df: DataFrame with price data and datetime index
        sentiment_data: dict from SentimentCollector
    
    Returns:
        DataFrame: Original data with sentiment features added
    """
    if sentiment_data is None:
        logger.warning("No sentiment data provided, using defaults")
        return add_default_sentiment_features(df)
    
    df = df.copy()
    
    # News sentiment
    news = sentiment_data.get('news', {})
    df['news_sentiment'] = news.get('sentiment', 0.0)
    df['news_article_count'] = news.get('article_count', 0)
    df['news_buzz'] = np.log1p(news.get('article_count', 0))  # Log scale for count
    
    # Social media sentiment
    reddit = sentiment_data.get('reddit', {})
    df['reddit_sentiment'] = reddit.get('sentiment', 0.0)
    df['reddit_mentions'] = reddit.get('mentions', 0)
    df['reddit_buzz'] = np.log1p(reddit.get('mentions', 0))
    
    # Economic indicators
    economic = sentiment_data.get('economic', {})
    df['vix'] = economic.get('vix', 20.0)
    df['unemployment_rate'] = economic.get('unemployment', 4.0)
    df['gdp_growth'] = economic.get('gdp_growth', 2.0)
    df['fed_rate'] = economic.get('fed_rate', 5.0)
    
    # Normalize VIX (fear index)
    df['vix_normalized'] = (df['vix'] - 20) / 10  # Center at 20, scale by 10
    df['market_fear'] = (df['vix'] > 25).astype(int)  # High fear flag
    
    # Google Trends
    trends = sentiment_data.get('trends', {})
    df['search_interest'] = trends.get('interest', 50.0)
    df['search_interest_normalized'] = (df['search_interest'] - 50) / 50
    
    # Composite sentiment
    df['composite_sentiment'] = sentiment_data.get('composite_sentiment', 0.0)
    
    # Sentiment strength (absolute value)
    df['sentiment_strength'] = np.abs(df['composite_sentiment'])
    
    # Sentiment direction
    df['sentiment_positive'] = (df['composite_sentiment'] > 0).astype(int)
    df['sentiment_negative'] = (df['composite_sentiment'] < 0).astype(int)
    
    # Combined buzz score (news + social)
    df['total_buzz'] = df['news_buzz'] + df['reddit_buzz']
    
    # Economic health composite
    # Lower unemployment + higher GDP = better economy
    df['economic_health'] = (
        -df['unemployment_rate'] / 10 +  # Normalize unemployment
        df['gdp_growth'] / 5  # Normalize GDP
    )
    
    logger.info(f"Added {18} sentiment features")
    
    return df

def add_default_sentiment_features(df):
    """
    Add default neutral sentiment features when no data available
    
    Args:
        df: DataFrame with price data
    
    Returns:
        DataFrame: Original data with default sentiment features
    """
    df = df.copy()
    
    # Set all sentiment features to neutral/default values
    df['news_sentiment'] = 0.0
    df['news_article_count'] = 0
    df['news_buzz'] = 0.0
    
    df['reddit_sentiment'] = 0.0
    df['reddit_mentions'] = 0
    df['reddit_buzz'] = 0.0
    
    df['vix'] = 20.0  # Historical average
    df['unemployment_rate'] = 4.0
    df['gdp_growth'] = 2.0
    df['fed_rate'] = 5.0
    df['vix_normalized'] = 0.0
    df['market_fear'] = 0
    
    df['search_interest'] = 50.0
    df['search_interest_normalized'] = 0.0
    
    df['composite_sentiment'] = 0.0
    df['sentiment_strength'] = 0.0
    df['sentiment_positive'] = 0
    df['sentiment_negative'] = 0
    
    df['total_buzz'] = 0.0
    df['economic_health'] = 0.0
    
    return df

def aggregate_sentiment_for_timeframe(sentiment_data_list):
    """
    Aggregate multiple sentiment data points (useful for intraday data)
    
    Args:
        sentiment_data_list: List of sentiment dicts from different times
    
    Returns:
        dict: Aggregated sentiment data
    """
    if not sentiment_data_list:
        return None
    
    # Average numerical values
    news_sentiments = [s['news']['sentiment'] for s in sentiment_data_list]
    reddit_sentiments = [s['reddit']['sentiment'] for s in sentiment_data_list]
    composite_sentiments = [s['composite_sentiment'] for s in sentiment_data_list]
    
    # Take max article count and mentions (cumulative)
    news_counts = [s['news']['article_count'] for s in sentiment_data_list]
    reddit_mentions = [s['reddit']['mentions'] for s in sentiment_data_list]
    
    # Economic indicators are typically constant for a day
    economic = sentiment_data_list[0]['economic']
    trends = sentiment_data_list[0]['trends']
    
    aggregated = {
        'news': {
            'sentiment': np.mean(news_sentiments),
            'article_count': max(news_counts)
        },
        'reddit': {
            'sentiment': np.mean(reddit_sentiments),
            'mentions': max(reddit_mentions)
        },
        'economic': economic,
        'trends': trends,
        'composite_sentiment': np.mean(composite_sentiments)
    }
    
    return aggregated

if __name__ == '__main__':
    # Test with sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    
    df = pd.DataFrame({
        'Close': 100 + np.random.randn(100).cumsum()
    }, index=dates)
    
    # Mock sentiment data
    sentiment_data = {
        'news': {'sentiment': 0.3, 'article_count': 5},
        'reddit': {'sentiment': -0.1, 'mentions': 12},
        'economic': {'vix': 18.5, 'unemployment': 3.8, 'gdp_growth': 2.5, 'fed_rate': 5.25},
        'trends': {'interest': 65.0},
        'composite_sentiment': 0.15
    }
    
    df_with_sentiment = add_sentiment_features(df, sentiment_data)
    
    print("\nSample data with sentiment features:")
    print(df_with_sentiment.head())
    print(f"\nSentiment feature columns:")
    sentiment_cols = [col for col in df_with_sentiment.columns if col != 'Close']
    print(sentiment_cols)

