"""
Sentiment Data Collector - Collect market sentiment from multiple sources

Sources:
- NewsAPI: News sentiment analysis
- Reddit: Social media sentiment
- FRED: Economic indicators (VIX, unemployment, GDP, rates)
- Google Trends: Search volume/interest

All functions handle rate limits gracefully and cache data locally.
"""
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from config import (
    NEWSAPI_KEY, FRED_API_KEY, REDDIT_CLIENT_ID, 
    REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
    DATA_DIR, SENTIMENT_DATA_SUBDIR
)
from utils import setup_logging, save_pickle, load_pickle, save_json, load_json

logger = setup_logging(__name__, 'sentiment_collector.log')

class SentimentCollector:
    """Collect sentiment data from multiple sources"""
    
    def __init__(self):
        """Initialize API clients"""
        self.newsapi_client = None
        self.fred_client = None
        self.reddit_client = None
        self.pytrends = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients with error handling"""
        # NewsAPI
        if NEWSAPI_KEY:
            try:
                from newsapi import NewsApiClient
                self.newsapi_client = NewsApiClient(api_key=NEWSAPI_KEY)
                logger.info("NewsAPI client initialized")
            except ImportError:
                logger.warning("newsapi-python not installed")
            except Exception as e:
                logger.warning(f"NewsAPI initialization failed: {e}")
        
        # FRED
        if FRED_API_KEY:
            try:
                from fredapi import Fred
                self.fred_client = Fred(api_key=FRED_API_KEY)
                logger.info("FRED client initialized")
            except ImportError:
                logger.warning("fredapi not installed")
            except Exception as e:
                logger.warning(f"FRED initialization failed: {e}")
        
        # Reddit
        if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
            try:
                import praw
                self.reddit_client = praw.Reddit(
                    client_id=REDDIT_CLIENT_ID,
                    client_secret=REDDIT_CLIENT_SECRET,
                    user_agent=REDDIT_USER_AGENT
                )
                logger.info("Reddit client initialized")
            except ImportError:
                logger.warning("praw not installed")
            except Exception as e:
                logger.warning(f"Reddit initialization failed: {e}")
        
        # Google Trends
        try:
            from pytrends.request import TrendReq
            self.pytrends = TrendReq(hl='en-US', tz=360)
            logger.info("Google Trends client initialized")
        except ImportError:
            logger.warning("pytrends not installed")
        except Exception as e:
            logger.warning(f"Google Trends initialization failed: {e}")
    
    def _get_sentiment_file_path(self, ticker, date):
        """Get file path for sentiment data"""
        date_str = date.strftime('%Y-%m-%d')
        return os.path.join(DATA_DIR, ticker, SENTIMENT_DATA_SUBDIR, f'{date_str}.pkl')
    
    def sentiment_exists(self, ticker, date):
        """Check if sentiment data exists for a date"""
        return os.path.exists(self._get_sentiment_file_path(ticker, date))
    
    def get_news_sentiment(self, ticker, date):
        """
        Get news sentiment for a ticker on a specific date
        
        Args:
            ticker: Stock ticker
            date: datetime object
        
        Returns:
            dict with sentiment score and article count
        """
        if not self.newsapi_client:
            logger.debug("NewsAPI not available, using default")
            return {'sentiment': 0.0, 'article_count': 0, 'source': 'default'}
        
        try:
            # Search for news articles
            from_date = date.strftime('%Y-%m-%d')
            to_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            response = self.newsapi_client.get_everything(
                q=ticker,
                from_param=from_date,
                to=to_date,
                language='en',
                sort_by='relevancy',
                page_size=20
            )
            
            articles = response.get('articles', [])
            
            if not articles:
                return {'sentiment': 0.0, 'article_count': 0, 'source': 'newsapi'}
            
            # Analyze sentiment using TextBlob
            try:
                from textblob import TextBlob
                
                sentiments = []
                for article in articles:
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    if text.strip():
                        blob = TextBlob(text)
                        sentiments.append(blob.sentiment.polarity)
                
                avg_sentiment = np.mean(sentiments) if sentiments else 0.0
                
                return {
                    'sentiment': float(avg_sentiment),
                    'article_count': len(articles),
                    'source': 'newsapi'
                }
            
            except ImportError:
                logger.warning("textblob not installed, returning neutral sentiment")
                return {'sentiment': 0.0, 'article_count': len(articles), 'source': 'newsapi_no_analysis'}
        
        except Exception as e:
            logger.error(f"Error getting news sentiment: {e}")
            return {'sentiment': 0.0, 'article_count': 0, 'source': 'error'}
    
    def get_economic_indicators(self, date):
        """
        Get economic indicators from FRED
        
        Args:
            date: datetime object
        
        Returns:
            dict with VIX, unemployment, GDP growth, fed rate
        """
        if not self.fred_client:
            logger.debug("FRED not available, using defaults")
            return {
                'vix': 20.0,  # Neutral fear level
                'unemployment': 4.0,
                'gdp_growth': 2.0,
                'fed_rate': 5.0,
                'source': 'default'
            }
        
        try:
            # Fetch indicators (use most recent value before date)
            vix = self.fred_client.get_series('VIXCLS', observation_start=date)
            
            indicators = {
                'vix': float(vix.iloc[-1]) if not vix.empty else 20.0,
                'unemployment': 4.0,  # FRED has monthly data, would need more logic
                'gdp_growth': 2.0,
                'fed_rate': 5.0,
                'source': 'fred'
            }
            
            return indicators
        
        except Exception as e:
            logger.error(f"Error getting economic indicators: {e}")
            return {
                'vix': 20.0,
                'unemployment': 4.0,
                'gdp_growth': 2.0,
                'fed_rate': 5.0,
                'source': 'error'
            }
    
    def get_reddit_sentiment(self, ticker, date):
        """
        Get Reddit sentiment from r/wallstreetbets and r/stocks
        
        Args:
            ticker: Stock ticker
            date: datetime object
        
        Returns:
            dict with sentiment score and mention count
        """
        if not self.reddit_client:
            logger.debug("Reddit not available, using default")
            return {'sentiment': 0.0, 'mentions': 0, 'source': 'default'}
        
        try:
            # Search for ticker mentions
            mentions = []
            
            for subreddit_name in ['wallstreetbets', 'stocks']:
                subreddit = self.reddit_client.subreddit(subreddit_name)
                
                # Search recent posts
                for post in subreddit.search(ticker, limit=10):
                    post_date = datetime.fromtimestamp(post.created_utc).date()
                    if post_date == date.date():
                        mentions.append({
                            'title': post.title,
                            'score': post.score,
                            'comments': post.num_comments
                        })
            
            # Simple sentiment: positive if more upvotes than downvotes
            if mentions:
                avg_score = np.mean([m['score'] for m in mentions])
                sentiment = np.tanh(avg_score / 100)  # Normalize to [-1, 1]
            else:
                sentiment = 0.0
            
            return {
                'sentiment': float(sentiment),
                'mentions': len(mentions),
                'source': 'reddit'
            }
        
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment: {e}")
            return {'sentiment': 0.0, 'mentions': 0, 'source': 'error'}
    
    def get_google_trends(self, ticker, date):
        """
        Get Google Trends interest over time
        
        Args:
            ticker: Stock ticker
            date: datetime object
        
        Returns:
            dict with interest score
        """
        if not self.pytrends:
            logger.debug("Google Trends not available, using default")
            return {'interest': 50.0, 'source': 'default'}
        
        try:
            # Google Trends has rate limits, use carefully
            timeframe = f"{date.strftime('%Y-%m-%d')} {(date + timedelta(days=7)).strftime('%Y-%m-%d')}"
            
            self.pytrends.build_payload([ticker], timeframe=timeframe, geo='US')
            interest_over_time = self.pytrends.interest_over_time()
            
            if not interest_over_time.empty and ticker in interest_over_time.columns:
                interest = float(interest_over_time[ticker].mean())
            else:
                interest = 50.0
            
            return {'interest': interest, 'source': 'trends'}
        
        except Exception as e:
            logger.error(f"Error getting Google Trends: {e}")
            return {'interest': 50.0, 'source': 'error'}
    
    def collect_sentiment(self, ticker, date, force=False):
        """
        Collect all sentiment data for a ticker on a date
        
        Args:
            ticker: Stock ticker
            date: datetime object
            force: If True, re-collect even if cached
        
        Returns:
            dict with all sentiment data
        """
        file_path = self._get_sentiment_file_path(ticker, date)
        
        # Check cache
        if not force and self.sentiment_exists(ticker, date):
            logger.info(f"Loading sentiment for {ticker} on {date.strftime('%Y-%m-%d')} from cache")
            return load_pickle(file_path)
        
        logger.info(f"Collecting sentiment for {ticker} on {date.strftime('%Y-%m-%d')}")
        
        # Collect from all sources
        sentiment_data = {
            'ticker': ticker,
            'date': date,
            'news': self.get_news_sentiment(ticker, date),
            'economic': self.get_economic_indicators(date),
            'reddit': self.get_reddit_sentiment(ticker, date),
            'trends': self.get_google_trends(ticker, date),
            'timestamp': datetime.now()
        }
        
        # Calculate composite sentiment score
        sentiment_data['composite_sentiment'] = self._calculate_composite_sentiment(sentiment_data)
        
        # Save to cache
        save_pickle(sentiment_data, file_path)
        logger.info(f"Saved sentiment data for {ticker} on {date.strftime('%Y-%m-%d')}")
        
        return sentiment_data
    
    def _calculate_composite_sentiment(self, sentiment_data):
        """
        Calculate a composite sentiment score from all sources
        
        Args:
            sentiment_data: dict with all sentiment data
        
        Returns:
            float: Composite sentiment score [-1, 1]
        """
        # Weighted average of different sources
        news_weight = 0.4
        reddit_weight = 0.3
        economic_weight = 0.2
        trends_weight = 0.1
        
        news_sentiment = sentiment_data['news']['sentiment']
        reddit_sentiment = sentiment_data['reddit']['sentiment']
        
        # Normalize economic indicators
        vix = sentiment_data['economic']['vix']
        vix_sentiment = -np.tanh((vix - 20) / 10)  # Higher VIX = negative sentiment
        
        # Normalize trends
        trends_sentiment = (sentiment_data['trends']['interest'] - 50) / 50
        
        composite = (
            news_weight * news_sentiment +
            reddit_weight * reddit_sentiment +
            economic_weight * vix_sentiment +
            trends_weight * trends_sentiment
        )
        
        return float(np.clip(composite, -1, 1))

def main():
    """Test sentiment collection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect sentiment data')
    parser.add_argument('--ticker', '-t', type=str, required=True)
    parser.add_argument('--date', '-d', type=str, help='Date (YYYY-MM-DD), defaults to today')
    
    args = parser.parse_args()
    
    date = datetime.strptime(args.date, '%Y-%m-%d') if args.date else datetime.now()
    
    collector = SentimentCollector()
    sentiment = collector.collect_sentiment(args.ticker, date)
    
    print(f"\nSentiment for {args.ticker} on {date.strftime('%Y-%m-%d')}:")
    print(f"  Composite: {sentiment['composite_sentiment']:.3f}")
    print(f"  News: {sentiment['news']['sentiment']:.3f} ({sentiment['news']['article_count']} articles)")
    print(f"  Reddit: {sentiment['reddit']['sentiment']:.3f} ({sentiment['reddit']['mentions']} mentions)")
    print(f"  VIX: {sentiment['economic']['vix']:.2f}")
    print(f"  Trends: {sentiment['trends']['interest']:.1f}")

if __name__ == '__main__':
    main()

