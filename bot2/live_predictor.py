"""
Live Predictor - Real-time stock price predictions

Fetches latest data, computes features, and makes predictions in real-time.
Includes error handling, retry logic, and logging.
"""
import time
from datetime import datetime, timedelta
import pandas as pd
from config import LOOKBACK_WINDOW, validate_config
from utils import setup_logging, is_market_open, Timer, filter_regular_hours
from data_downloader import DataDownloader
from sentiment_collector import SentimentCollector
from predictor import StockPredictor
from technical_indicators import add_all_indicators
from temporal_features import add_all_temporal_features
from sentiment_features import add_sentiment_features

logger = setup_logging(__name__, 'live_predictor.log')

class LivePredictor:
    """Make real-time predictions with live data"""
    
    def __init__(self, ticker):
        """
        Initialize live predictor
        
        Args:
            ticker: Stock ticker symbol
        """
        self.ticker = ticker
        self.predictor = StockPredictor(ticker)
        self.downloader = DataDownloader()
        self.sentiment_collector = SentimentCollector()
        
        # Cache for features (update periodically, not every minute)
        self.sentiment_cache = None
        self.sentiment_cache_time = None
        self.sentiment_cache_ttl = 3600  # 1 hour TTL
        
        # Initialize predictor
        try:
            self.predictor.initialize()
            logger.info(f"Live predictor initialized for {ticker}")
        except FileNotFoundError as e:
            logger.error(f"Failed to initialize predictor: {e}")
            raise
    
    def get_latest_data(self, lookback_days=5):
        """
        Fetch latest data for the ticker
        
        Args:
            lookback_days: Number of days to fetch for context
        
        Returns:
            DataFrame with raw price data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Download recent data if not cached
        logger.info(f"Fetching latest data for {self.ticker}")
        self.downloader.download_range(self.ticker, start_date, end_date)
        
        # Load downloaded data
        from utils import get_trading_days, load_pickle
        from config import DATA_DIR, RAW_DATA_SUBDIR
        import os
        
        trading_days = get_trading_days(start_date, end_date)
        dfs = []
        
        for date in trading_days:
            file_path = os.path.join(
                DATA_DIR, self.ticker, RAW_DATA_SUBDIR,
                f'{date.strftime("%Y-%m-%d")}.pkl'
            )
            if os.path.exists(file_path):
                df = load_pickle(file_path)
                if df is not None and not df.empty:
                    dfs.append(df)
        
        if not dfs:
            logger.error("No data available")
            return None
        
        # Combine all data
        combined_df = pd.concat(dfs, axis=0)
        combined_df = combined_df.sort_index()
        
        # Filter to regular market hours only
        original_len = len(combined_df)
        combined_df = filter_regular_hours(combined_df)
        filtered_count = original_len - len(combined_df)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} pre/after-market rows")
        
        logger.info(f"Loaded {len(combined_df)} rows of regular market hours data")
        return combined_df
    
    def get_sentiment(self, force_refresh=False):
        """
        Get sentiment data (cached for 1 hour)
        
        Args:
            force_refresh: Force refresh cache
        
        Returns:
            dict with sentiment data
        """
        now = datetime.now()
        
        # Check cache
        if not force_refresh and self.sentiment_cache is not None:
            if self.sentiment_cache_time is not None:
                age = (now - self.sentiment_cache_time).total_seconds()
                if age < self.sentiment_cache_ttl:
                    logger.debug("Using cached sentiment data")
                    return self.sentiment_cache
        
        # Fetch new sentiment
        logger.info("Fetching fresh sentiment data")
        sentiment = self.sentiment_collector.collect_sentiment(self.ticker, now)
        
        self.sentiment_cache = sentiment
        self.sentiment_cache_time = now
        
        return sentiment
    
    def compute_features(self, df):
        """
        Compute all features for prediction
        
        Args:
            df: DataFrame with raw OHLCV data
        
        Returns:
            DataFrame with all features
        """
        logger.info("Computing features...")
        
        with Timer("Feature computation"):
            # Technical indicators
            df = add_all_indicators(df)
            
            # Temporal features
            df = add_all_temporal_features(df, self.ticker)
            
            # Sentiment features
            sentiment = self.get_sentiment()
            df = add_sentiment_features(df, sentiment)
            
            # Drop NaN values
            df = df.dropna()
        
        logger.info(f"Features computed: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def make_prediction(self, retry_count=3):
        """
        Make a prediction with latest data
        
        Args:
            retry_count: Number of retries on failure
        
        Returns:
            dict with prediction or None on failure
        """
        for attempt in range(retry_count):
            try:
                # Get latest data
                df = self.get_latest_data()
                if df is None or len(df) < LOOKBACK_WINDOW:
                    logger.warning("Insufficient data for prediction")
                    return None
                
                # Compute features
                df = self.compute_features(df)
                
                if df is None or len(df) < LOOKBACK_WINDOW:
                    logger.warning("Insufficient data after feature computation")
                    return None
                
                # Make prediction
                prediction = self.predictor.predict(df)
                
                # Add timestamp
                prediction['prediction_time'] = datetime.now()
                
                logger.info(f"Prediction: {prediction['predicted_direction']} "
                          f"with {prediction['confidence']:.1%} confidence")
                
                return prediction
            
            except Exception as e:
                logger.error(f"Prediction failed (attempt {attempt+1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(5)  # Wait before retry
                else:
                    return None
    
    def run_live(self, interval_minutes=1, max_predictions=None):
        """
        Run live predictions continuously
        
        Args:
            interval_minutes: Minutes between predictions
            max_predictions: Max number of predictions (None = infinite)
        """
        logger.info(f"Starting live predictions for {self.ticker}")
        logger.info(f"Interval: {interval_minutes} minute(s)")
        
        prediction_count = 0
        
        try:
            while True:
                # Check if market is open
                if not is_market_open():
                    logger.info("Market is closed, waiting...")
                    time.sleep(300)  # Wait 5 minutes
                    continue
                
                # Make prediction
                prediction = self.make_prediction()
                
                if prediction:
                    # Display prediction
                    print(self.predictor.format_prediction(prediction))
                    
                    # Log to file
                    self.log_prediction(prediction)
                    
                    prediction_count += 1
                    
                    # Check if max reached
                    if max_predictions and prediction_count >= max_predictions:
                        logger.info(f"Reached max predictions ({max_predictions})")
                        break
                
                # Wait for next interval
                logger.info(f"Waiting {interval_minutes} minute(s) for next prediction...")
                time.sleep(interval_minutes * 60)
        
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
    
    def log_prediction(self, prediction):
        """
        Log prediction to file
        
        Args:
            prediction: dict from make_prediction
        """
        from utils import save_json
        from config import LOGS_DIR
        import os
        
        log_file = os.path.join(LOGS_DIR, f'{self.ticker}_predictions.jsonl')
        
        # Append to JSONL file
        with open(log_file, 'a') as f:
            import json
            f.write(json.dumps(prediction, default=str) + '\n')

def main():
    """Command-line interface for live predictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live stock price predictions')
    parser.add_argument('--ticker', '-t', type=str, required=True,
                       help='Stock ticker symbol')
    parser.add_argument('--interval', '-i', type=int, default=1,
                       help='Minutes between predictions (default: 1)')
    parser.add_argument('--max-predictions', type=int,
                       help='Maximum number of predictions (default: infinite)')
    parser.add_argument('--once', action='store_true',
                       help='Make one prediction and exit')
    
    args = parser.parse_args()
    
    # Validate configuration
    try:
        validate_config()
    except ValueError as e:
        logger.error(str(e))
        return
    
    # Initialize live predictor
    try:
        live = LivePredictor(args.ticker)
    except Exception as e:
        print(f"Failed to initialize: {e}")
        print(f"Make sure you've trained a model for {args.ticker} first.")
        return
    
    # Run predictions
    if args.once:
        prediction = live.make_prediction()
        if prediction:
            print(live.predictor.format_prediction(prediction))
    else:
        live.run_live(
            interval_minutes=args.interval,
            max_predictions=args.max_predictions
        )

if __name__ == '__main__':
    main()

