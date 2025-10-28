"""
Feature Pipeline - Orchestrate all feature engineering

Loads raw data, computes technical indicators, temporal features,
and sentiment features, then saves processed data for training.
"""
import os
from datetime import datetime
import pandas as pd
import numpy as np
from config import DATA_DIR, RAW_DATA_SUBDIR, PROCESSED_DATA_SUBDIR
from utils import setup_logging, save_pickle, load_pickle, Timer, print_dataframe_info, filter_regular_hours
from technical_indicators import add_all_indicators
from temporal_features import add_all_temporal_features
from sentiment_features import add_sentiment_features
from data_downloader import DataDownloader
from sentiment_collector import SentimentCollector

logger = setup_logging(__name__, 'feature_pipeline.log')

class FeaturePipeline:
    """Orchestrate feature engineering pipeline"""
    
    def __init__(self):
        """Initialize pipeline components"""
        self.sentiment_collector = SentimentCollector()
    
    def _get_raw_data_path(self, ticker, date):
        """Get path to raw data file"""
        date_str = date.strftime('%Y-%m-%d')
        return os.path.join(DATA_DIR, ticker, RAW_DATA_SUBDIR, f'{date_str}.pkl')
    
    def _get_processed_data_path(self, ticker, date):
        """Get path to processed data file"""
        date_str = date.strftime('%Y-%m-%d')
        return os.path.join(DATA_DIR, ticker, PROCESSED_DATA_SUBDIR, f'{date_str}.pkl')
    
    def processed_data_exists(self, ticker, date):
        """Check if processed data already exists"""
        return os.path.exists(self._get_processed_data_path(ticker, date))
    
    def load_raw_data(self, ticker, date):
        """
        Load raw OHLCV data for a date
        
        Args:
            ticker: Stock ticker
            date: datetime object
        
        Returns:
            DataFrame or None if not available
        """
        path = self._get_raw_data_path(ticker, date)
        
        if not os.path.exists(path):
            logger.warning(f"Raw data not found for {ticker} on {date.strftime('%Y-%m-%d')}")
            return None
        
        return load_pickle(path)
    
    def process_day(self, ticker, date, force=False):
        """
        Process a single day of data through feature pipeline
        
        Args:
            ticker: Stock ticker
            date: datetime object
            force: If True, reprocess even if already exists
        
        Returns:
            DataFrame with all features or None
        """
        # Check if already processed
        processed_path = self._get_processed_data_path(ticker, date)
        if not force and self.processed_data_exists(ticker, date):
            logger.info(f"Loading processed data for {ticker} on {date.strftime('%Y-%m-%d')}")
            return load_pickle(processed_path)
        
        logger.info(f"Processing {ticker} for {date.strftime('%Y-%m-%d')}")
        
        # Load raw data
        df = self.load_raw_data(ticker, date)
        if df is None or df.empty:
            logger.warning(f"No raw data to process for {ticker} on {date.strftime('%Y-%m-%d')}")
            return None
        
        try:
            with Timer(f"Feature engineering for {date.strftime('%Y-%m-%d')}"):
                # Filter to regular market hours only (9:30 AM - 4:00 PM)
                original_len = len(df)
                df = filter_regular_hours(df)
                filtered_count = original_len - len(df)
                if filtered_count > 0:
                    logger.info(f"Filtered out {filtered_count} pre/after-market rows, keeping {len(df)} regular hours")
                
                # Add technical indicators
                df = add_all_indicators(df)
                
                # Add temporal features
                df = add_all_temporal_features(df, ticker)
                
                # Get and add sentiment features
                sentiment_data = self.sentiment_collector.collect_sentiment(ticker, date)
                df = add_sentiment_features(df, sentiment_data)
                
                # Drop any rows with NaN values (from indicator calculations)
                original_len = len(df)
                df = df.dropna()
                dropped = original_len - len(df)
                
                if dropped > 0:
                    logger.info(f"Dropped {dropped} rows with NaN values")
                
                if df.empty:
                    logger.warning(f"All data dropped for {ticker} on {date.strftime('%Y-%m-%d')}")
                    return None
                
                # Save processed data
                save_pickle(df, processed_path)
                logger.info(f"Saved processed data: {len(df)} rows, {len(df.columns)} features")
                
                return df
        
        except Exception as e:
            logger.error(f"Error processing {ticker} for {date.strftime('%Y-%m-%d')}: {e}")
            return None
    
    def process_date_range(self, ticker, start_date, end_date, force=False):
        """
        Process multiple days of data
        
        Args:
            ticker: Stock ticker
            start_date: datetime object
            end_date: datetime object
            force: If True, reprocess existing files
        
        Returns:
            int: Number of days successfully processed
        """
        from utils import get_trading_days
        
        trading_days = get_trading_days(start_date, end_date)
        logger.info(f"Processing {len(trading_days)} trading days for {ticker}")
        
        success_count = 0
        for date in trading_days:
            df = self.process_day(ticker, date, force=force)
            if df is not None and not df.empty:
                success_count += 1
        
        logger.info(f"Successfully processed {success_count}/{len(trading_days)} days")
        return success_count
    
    def load_processed_data_range(self, ticker, start_date, end_date):
        """
        Load all processed data for a date range and concatenate
        
        Args:
            ticker: Stock ticker
            start_date: datetime object
            end_date: datetime object
        
        Returns:
            DataFrame: Concatenated data from all days
        """
        from utils import get_trading_days
        
        trading_days = get_trading_days(start_date, end_date)
        logger.info(f"Loading processed data for {len(trading_days)} days")
        
        dfs = []
        for date in trading_days:
            processed_path = self._get_processed_data_path(ticker, date)
            if os.path.exists(processed_path):
                df = load_pickle(processed_path)
                if df is not None and not df.empty:
                    dfs.append(df)
            else:
                logger.warning(f"Processed data not found for {date.strftime('%Y-%m-%d')}")
        
        if not dfs:
            logger.error("No processed data found in date range")
            return None
        
        # Concatenate all data
        combined_df = pd.concat(dfs, axis=0)
        combined_df = combined_df.sort_index()
        
        logger.info(f"Loaded {len(combined_df)} rows across {len(dfs)} days")
        logger.info(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        logger.info(f"Total features: {len(combined_df.columns)}")
        
        return combined_df
    
    def get_feature_names(self, ticker, sample_date=None):
        """
        Get list of all feature names from processed data
        
        Args:
            ticker: Stock ticker
            sample_date: Optional date to load sample from
        
        Returns:
            list: Feature column names
        """
        if sample_date is None:
            # Find any processed file
            processed_dir = os.path.join(DATA_DIR, ticker, PROCESSED_DATA_SUBDIR)
            if not os.path.exists(processed_dir):
                logger.error(f"No processed data directory for {ticker}")
                return []
            
            files = [f for f in os.listdir(processed_dir) if f.endswith('.pkl')]
            if not files:
                logger.error(f"No processed data files for {ticker}")
                return []
            
            sample_file = os.path.join(processed_dir, files[0])
            df = load_pickle(sample_file)
        else:
            df = load_pickle(self._get_processed_data_path(ticker, sample_date))
        
        if df is None:
            return []
        
        # Exclude OHLCV columns (keep only engineered features)
        base_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_columns = [col for col in df.columns if col not in base_columns]
        
        return feature_columns

def main():
    """Command-line interface for feature pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature engineering pipeline')
    parser.add_argument('--ticker', '-t', type=str, required=True)
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--force', '-f', action='store_true', help='Force reprocessing')
    parser.add_argument('--show-sample', action='store_true', help='Show sample of processed data')
    
    args = parser.parse_args()
    
    start = datetime.strptime(args.start_date, '%Y-%m-%d')
    end = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    pipeline = FeaturePipeline()
    
    # Process data
    success_count = pipeline.process_date_range(args.ticker, start, end, force=args.force)
    
    print(f"\n{'='*60}")
    print(f"Processing complete: {success_count} days processed")
    print(f"{'='*60}\n")
    
    # Show sample if requested
    if args.show_sample and success_count > 0:
        df = pipeline.load_processed_data_range(args.ticker, start, end)
        if df is not None:
            print_dataframe_info(df, f"{args.ticker} Processed Data")
            
            # Show feature breakdown
            features = pipeline.get_feature_names(args.ticker)
            print(f"\nTotal engineered features: {len(features)}")
            print("\nFeature names:")
            for i, feat in enumerate(features, 1):
                print(f"  {i:2d}. {feat}")

if __name__ == '__main__':
    main()

