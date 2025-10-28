"""
Data Manager - Track and manage downloaded data

Maintains metadata about what data has been downloaded,
identifies gaps, and orchestrates bulk downloads.
"""
import os
from datetime import datetime, timedelta
from config import DATA_DIR, RAW_DATA_SUBDIR
from utils import (
    setup_logging, get_trading_days, validate_ticker,
    save_json, load_json
)

logger = setup_logging(__name__, 'data_manager.log')

class DataManager:
    """Manage and track downloaded data"""
    
    def __init__(self):
        """Initialize data manager"""
        self.metadata = {}
    
    def _get_metadata_path(self, ticker):
        """Get metadata file path for a ticker"""
        return os.path.join(DATA_DIR, ticker, 'metadata.json')
    
    def load_metadata(self, ticker):
        """
        Load metadata for a ticker
        
        Args:
            ticker: Stock ticker
        
        Returns:
            dict: Metadata
        """
        ticker = validate_ticker(ticker)
        metadata_path = self._get_metadata_path(ticker)
        
        metadata = load_json(metadata_path)
        if metadata is None:
            metadata = {
                'ticker': ticker,
                'downloaded_dates': [],
                'first_date': None,
                'last_date': None,
                'total_days': 0,
                'last_updated': None
            }
        
        self.metadata[ticker] = metadata
        return metadata
    
    def save_metadata(self, ticker):
        """
        Save metadata for a ticker
        
        Args:
            ticker: Stock ticker
        """
        ticker = validate_ticker(ticker)
        if ticker not in self.metadata:
            return
        
        metadata_path = self._get_metadata_path(ticker)
        self.metadata[ticker]['last_updated'] = datetime.now().isoformat()
        save_json(self.metadata[ticker], metadata_path)
        logger.info(f"Saved metadata for {ticker}")
    
    def scan_downloaded_data(self, ticker):
        """
        Scan directory to find all downloaded data files
        
        Args:
            ticker: Stock ticker
        
        Returns:
            list: List of datetime objects for available dates
        """
        ticker = validate_ticker(ticker)
        data_dir = os.path.join(DATA_DIR, ticker, RAW_DATA_SUBDIR)
        
        if not os.path.exists(data_dir):
            logger.warning(f"No data directory for {ticker}")
            return []
        
        dates = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.pkl'):
                try:
                    date_str = filename.replace('.pkl', '')
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    dates.append(date)
                except ValueError:
                    logger.warning(f"Invalid filename format: {filename}")
        
        dates.sort()
        
        # Update metadata
        metadata = self.load_metadata(ticker)
        metadata['downloaded_dates'] = [d.strftime('%Y-%m-%d') for d in dates]
        metadata['first_date'] = dates[0].strftime('%Y-%m-%d') if dates else None
        metadata['last_date'] = dates[-1].strftime('%Y-%m-%d') if dates else None
        metadata['total_days'] = len(dates)
        self.save_metadata(ticker)
        
        logger.info(f"Found {len(dates)} data files for {ticker}")
        return dates
    
    def find_missing_dates(self, ticker, start_date, end_date):
        """
        Find missing trading days in a date range
        
        Args:
            ticker: Stock ticker
            start_date: datetime object
            end_date: datetime object
        
        Returns:
            list: List of missing datetime objects
        """
        ticker = validate_ticker(ticker)
        
        # Get all downloaded dates
        downloaded = self.scan_downloaded_data(ticker)
        downloaded_set = set(d.date() for d in downloaded)
        
        # Get expected trading days
        expected = get_trading_days(start_date, end_date)
        expected_set = set(d.date() for d in expected)
        
        # Find missing
        missing = expected_set - downloaded_set
        missing_dates = [datetime.combine(d, datetime.min.time()) for d in sorted(missing)]
        
        logger.info(f"Found {len(missing_dates)} missing dates for {ticker}")
        return missing_dates
    
    def get_data_stats(self, ticker):
        """
        Get statistics about downloaded data
        
        Args:
            ticker: Stock ticker
        
        Returns:
            dict: Statistics
        """
        metadata = self.load_metadata(ticker)
        
        if metadata['total_days'] == 0:
            return {
                'ticker': ticker,
                'status': 'No data downloaded',
                'total_days': 0
            }
        
        first_date = datetime.fromisoformat(metadata['first_date'])
        last_date = datetime.fromisoformat(metadata['last_date'])
        date_range = (last_date - first_date).days
        
        # Calculate completeness
        expected_days = len(get_trading_days(first_date, last_date))
        completeness = (metadata['total_days'] / expected_days * 100) if expected_days > 0 else 0
        
        return {
            'ticker': ticker,
            'first_date': metadata['first_date'],
            'last_date': metadata['last_date'],
            'total_days': metadata['total_days'],
            'date_range_days': date_range,
            'completeness': f"{completeness:.1f}%",
            'last_updated': metadata['last_updated']
        }
    
    def print_data_summary(self, ticker):
        """
        Print a summary of downloaded data
        
        Args:
            ticker: Stock ticker
        """
        stats = self.get_data_stats(ticker)
        
        print(f"\n{'='*50}")
        print(f"Data Summary: {ticker}")
        print(f"{'='*50}")
        
        if stats['total_days'] == 0:
            print("  No data downloaded yet")
        else:
            print(f"  Date range: {stats['first_date']} to {stats['last_date']}")
            print(f"  Total days: {stats['total_days']}")
            print(f"  Span: {stats['date_range_days']} calendar days")
            print(f"  Completeness: {stats['completeness']}")
            print(f"  Last updated: {stats['last_updated']}")
        
        print(f"{'='*50}\n")
    
    def bulk_download(self, ticker, start_date, end_date, downloader, force=False):
        """
        Orchestrate bulk download with progress tracking
        
        Args:
            ticker: Stock ticker
            start_date: datetime object
            end_date: datetime object
            downloader: DataDownloader instance
            force: If True, re-download existing files
        
        Returns:
            int: Number of days successfully downloaded
        """
        ticker = validate_ticker(ticker)
        
        # Find missing dates if not forcing
        if not force:
            missing_dates = self.find_missing_dates(ticker, start_date, end_date)
            if not missing_dates:
                logger.info(f"No missing data for {ticker} in date range")
                return 0
            dates_to_download = missing_dates
        else:
            dates_to_download = get_trading_days(start_date, end_date)
        
        logger.info(f"Starting bulk download: {len(dates_to_download)} days")
        
        success_count = 0
        for date in dates_to_download:
            df = downloader.download_day(ticker, date, force=force)
            if df is not None and not df.empty:
                success_count += 1
        
        # Update metadata
        self.scan_downloaded_data(ticker)
        
        return success_count

def main():
    """Test data manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage downloaded data')
    parser.add_argument('--ticker', '-t', type=str, required=True)
    parser.add_argument('--scan', action='store_true', help='Scan and show summary')
    parser.add_argument('--check-missing', action='store_true', help='Check for missing dates')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    manager = DataManager()
    
    if args.scan:
        manager.scan_downloaded_data(args.ticker)
        manager.print_data_summary(args.ticker)
    
    if args.check_missing and args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, '%Y-%m-%d')
        end = datetime.strptime(args.end_date, '%Y-%m-%d')
        missing = manager.find_missing_dates(args.ticker, start, end)
        
        print(f"\nMissing dates for {args.ticker}:")
        for date in missing:
            print(f"  {date.strftime('%Y-%m-%d')}")

if __name__ == '__main__':
    main()

