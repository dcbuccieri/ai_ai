"""
Data Downloader - Download OHLCV data from Polygon.io

Handles:
- Date-based caching (skip existing files)
- Rate limiting (5 calls/min for free tier)
- Data storage in organized directory structure
- Error handling and retry logic
"""
import os
import time
import argparse
from datetime import datetime, timedelta
import pandas as pd
import pytz
from polygon import RESTClient
from config import (
    POLYGON_API_KEY, DATA_DIR, RAW_DATA_SUBDIR,
    POLYGON_RATE_LIMIT, validate_config
)
from utils import (
    setup_logging, is_market_open, get_trading_days,
    save_pickle, load_pickle, validate_ticker, Timer
)

logger = setup_logging(__name__, 'data_downloader.log')

class DataDownloader:
    """Download and cache stock data from Polygon.io"""
    
    def __init__(self, api_key=None):
        """
        Initialize Polygon client
        
        Args:
            api_key: Polygon API key (defaults to config)
        """
        self.api_key = api_key or POLYGON_API_KEY
        if not self.api_key:
            raise ValueError("Polygon API key not set. Add to .env file.")
        
        self.client = RESTClient(self.api_key)
        self.rate_limit_delay = 60 / POLYGON_RATE_LIMIT  # seconds between calls
        self.last_call_time = 0
        
        logger.info("DataDownloader initialized")
    
    def _rate_limit_wait(self):
        """Wait if necessary to respect rate limits"""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - elapsed
            logger.debug(f"Rate limit wait: {wait_time:.2f}s")
            time.sleep(wait_time)
        self.last_call_time = time.time()
    
    def _get_file_path(self, ticker, date):
        """
        Get file path for a specific ticker and date
        
        Args:
            ticker: Stock ticker symbol
            date: datetime object
        
        Returns:
            File path string
        """
        date_str = date.strftime('%Y-%m-%d')
        return os.path.join(DATA_DIR, ticker, RAW_DATA_SUBDIR, f'{date_str}.pkl')
    
    def data_exists(self, ticker, date):
        """
        Check if data already exists for a given date
        
        Args:
            ticker: Stock ticker symbol
            date: datetime object
        
        Returns:
            bool: True if data file exists
        """
        return os.path.exists(self._get_file_path(ticker, date))
    
    def download_day(self, ticker, date, force=False):
        """
        Download data for a single day
        
        Args:
            ticker: Stock ticker symbol
            date: datetime object
            force: If True, re-download even if file exists
        
        Returns:
            DataFrame with OHLCV data or None if no data available
        """
        ticker = validate_ticker(ticker)
        file_path = self._get_file_path(ticker, date)
        
        # Check if already downloaded
        if not force and self.data_exists(ticker, date):
            logger.info(f"Loading {ticker} for {date.strftime('%Y-%m-%d')} from cache")
            return load_pickle(file_path)
        
        # Skip weekends
        if date.weekday() >= 5:
            logger.info(f"Skipping {date.strftime('%Y-%m-%d')} (weekend)")
            return None
        
        # Download from Polygon
        logger.info(f"Downloading {ticker} for {date.strftime('%Y-%m-%d')} from Polygon.io")
        
        try:
            self._rate_limit_wait()
            
            # Format dates for API
            from_date = date.strftime('%Y-%m-%d')
            to_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Fetch 1-minute aggregates
            aggs = []
            for agg in self.client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan='minute',
                from_=from_date,
                to=to_date,
                limit=50000
            ):
                aggs.append({
                    'timestamp': pd.to_datetime(agg.timestamp, unit='ms'),
                    'Open': agg.open,
                    'High': agg.high,
                    'Low': agg.low,
                    'Close': agg.close,
                    'Volume': agg.volume,
                })
            
            if not aggs:
                logger.warning(f"No data available for {ticker} on {date.strftime('%Y-%m-%d')}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(aggs)
            df.set_index('timestamp', inplace=True)
            
            # Convert to Eastern Time
            df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
            
            # Filter to specific date
            target_date = date.date()
            df = df[df.index.date == target_date]
            
            if df.empty:
                logger.warning(f"No data for {ticker} on {date.strftime('%Y-%m-%d')} after filtering")
                return None
            
            # Save to cache
            save_pickle(df, file_path)
            logger.info(f"Saved {len(df)} records for {ticker} on {date.strftime('%Y-%m-%d')}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading {ticker} for {date.strftime('%Y-%m-%d')}: {e}")
            return None
    
    def download_bulk(self, ticker, start_date, end_date, timeframe='1m'):
        """
        Download data in bulk (week or month chunks) and split into daily files
        
        Args:
            ticker: Stock ticker symbol
            start_date: datetime object
            end_date: datetime object
            timeframe: Data timeframe ('1m', '5m', etc.)
        
        Returns:
            DataFrame with all data or None if failed
        """
        ticker = validate_ticker(ticker)
        
        logger.info(f"Bulk downloading {ticker} from {start_date.strftime('%Y-%m-%d')} "
                   f"to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            self._rate_limit_wait()
            
            # Format dates for API
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            # Fetch 1-minute aggregates
            aggs = []
            for agg in self.client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan='minute',
                from_=from_date,
                to=to_date,
                limit=50000
            ):
                aggs.append({
                    'timestamp': pd.to_datetime(agg.timestamp, unit='ms'),
                    'Open': agg.open,
                    'High': agg.high,
                    'Low': agg.low,
                    'Close': agg.close,
                    'Volume': agg.volume,
                })
            
            if not aggs:
                logger.warning(f"No data available for {ticker} in date range")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(aggs)
            df.set_index('timestamp', inplace=True)
            
            # Convert to Eastern Time
            df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
            
            logger.info(f"Bulk download successful: {len(df)} total rows")
            return df
            
        except Exception as e:
            logger.error(f"Bulk download failed: {e}")
            return None
    
    def split_and_save_bulk(self, ticker, df):
        """
        Split bulk downloaded data into daily files
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with data for multiple days
        
        Returns:
            int: Number of days saved
        """
        if df is None or df.empty:
            return 0
        
        # Group by date
        dates = df.index.date
        unique_dates = sorted(set(dates))
        
        saved_count = 0
        for date in unique_dates:
            # Filter to this date
            mask = df.index.date == date
            day_df = df[mask]
            
            if not day_df.empty:
                # Save to daily file
                date_obj = datetime.combine(date, datetime.min.time())
                file_path = self._get_file_path(ticker, date_obj)
                save_pickle(day_df, file_path)
                saved_count += 1
                logger.info(f"Saved {len(day_df)} records for {ticker} on {date.strftime('%Y-%m-%d')}")
        
        return saved_count
    
    def download_range(self, ticker, start_date, end_date, force=False, timeframe='1m'):
        """
        Download data for a date range using optimized bulk downloads
        
        Args:
            ticker: Stock ticker symbol
            start_date: datetime object
            end_date: datetime object
            force: If True, re-download existing files
            timeframe: Data timeframe ('1m' uses weekly chunks, others monthly)
        
        Returns:
            int: Number of days successfully downloaded
        """
        ticker = validate_ticker(ticker)
        trading_days = get_trading_days(start_date, end_date)
        
        # Check which days are already downloaded (if not forcing)
        if not force:
            missing_days = [d for d in trading_days if not self.data_exists(ticker, d)]
            if not missing_days:
                logger.info(f"All data already downloaded for {ticker}")
                return len(trading_days)
            logger.info(f"{len(missing_days)} days need downloading (out of {len(trading_days)})")
        else:
            missing_days = trading_days
        
        if not missing_days:
            return 0
        
        # Determine chunk size based on timeframe
        # 1-minute: weekly chunks (5 days * 650 min/day = 3,250 < 50,000)
        # 5+ minute: monthly chunks (20 days * 650 min/day = 13,000 < 50,000)
        chunk_days = 5 if timeframe == '1m' else 20  # ~1 week or ~1 month
        
        logger.info(f"Downloading {ticker} from {start_date.strftime('%Y-%m-%d')} "
                   f"to {end_date.strftime('%Y-%m-%d')} ({len(missing_days)} days) "
                   f"using {chunk_days}-day chunks")
        
        total_saved = 0
        failed_chunks = []
        
        with Timer(f"Bulk download {ticker} data"):
            # Process in chunks
            current = start_date
            while current <= end_date:
                chunk_end = min(current + timedelta(days=chunk_days), end_date)
                
                # Try bulk download for this chunk
                try:
                    df = self.download_bulk(ticker, current, chunk_end, timeframe)
                    
                    if df is not None and not df.empty:
                        # Split and save to daily files
                        saved = self.split_and_save_bulk(ticker, df)
                        total_saved += saved
                        logger.info(f"Chunk {current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}: "
                                  f"{saved} days saved")
                    else:
                        # Bulk download failed, fall back to day-by-day
                        logger.warning(f"Bulk download failed for chunk, trying day-by-day")
                        chunk_days_list = get_trading_days(current, chunk_end)
                        for date in chunk_days_list:
                            df_day = self.download_day(ticker, date, force=force)
                            if df_day is not None and not df_day.empty:
                                total_saved += 1
                
                except Exception as e:
                    error_msg = f"Failed to download {current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}: {e}"
                    logger.error(error_msg)
                    failed_chunks.append(error_msg)
                    
                    # Try day-by-day as fallback
                    logger.info("Attempting day-by-day fallback for failed chunk")
                    try:
                        chunk_days_list = get_trading_days(current, chunk_end)
                        for date in chunk_days_list:
                            try:
                                df_day = self.download_day(ticker, date, force=force)
                                if df_day is not None and not df_day.empty:
                                    total_saved += 1
                            except Exception as day_error:
                                logger.error(f"Day-by-day also failed for {date.strftime('%Y-%m-%d')}: {day_error}")
                    except Exception as fallback_error:
                        logger.error(f"Fallback failed: {fallback_error}")
                
                # Move to next chunk
                current = chunk_end + timedelta(days=1)
        
        logger.info(f"Successfully downloaded {total_saved}/{len(missing_days)} days")
        
        # Report failures prominently at the end
        if failed_chunks:
            print("\n" + "="*70)
            print("*** DOWNLOAD FAILURES DETECTED ***")
            print("="*70)
            for error in failed_chunks:
                print(f"  - {error}")
            print("="*70)
            print(f"Successfully saved: {total_saved} days")
            print(f"Failed chunks: {len(failed_chunks)}")
            print("="*70 + "\n")
        
        return total_saved
    
    def download_last_n_days(self, ticker, n_days, force=False):
        """
        Download last N trading days
        
        Args:
            ticker: Stock ticker symbol
            n_days: Number of days to download
            force: If True, re-download existing files
        
        Returns:
            int: Number of days successfully downloaded
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=n_days * 2)  # Extra buffer for weekends
        
        return self.download_range(ticker, start_date, end_date, force=force)

def main():
    """Command-line interface for data downloader"""
    parser = argparse.ArgumentParser(description='Download stock data from Polygon.io')
    parser.add_argument('--ticker', '-t', type=str, required=True,
                       help='Stock ticker symbol (e.g., AAPL, TSLA)')
    parser.add_argument('--days', '-d', type=int, default=7,
                       help='Number of days to download (default: 7)')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD format)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD format)')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force re-download of existing files')
    
    args = parser.parse_args()
    
    # Validate configuration
    try:
        validate_config()
    except ValueError as e:
        logger.error(str(e))
        return
    
    # Initialize downloader
    downloader = DataDownloader()
    
    # Download data
    if args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, '%Y-%m-%d')
        end = datetime.strptime(args.end_date, '%Y-%m-%d')
        downloader.download_range(args.ticker, start, end, force=args.force)
    else:
        downloader.download_last_n_days(args.ticker, args.days, force=args.force)
    
    logger.info("Download complete!")

if __name__ == '__main__':
    main()

