"""
System Test - Verify all components are working

Tests:
1. Configuration validity
2. API connections
3. Data download
4. Feature computation
5. Data preparation
6. Model can be built
"""
import sys
from datetime import datetime, timedelta
from config import validate_config
from utils import setup_logging

logger = setup_logging(__name__)

def test_config():
    """Test configuration"""
    print("\n" + "="*60)
    print("Test 1: Configuration")
    print("="*60)
    
    try:
        validate_config()
        print("[OK] Configuration valid")
        return True
    except Exception as e:
        print(f"[FAIL] Configuration error: {e}")
        return False

def test_api_connections():
    """Test API connections"""
    print("\n" + "="*60)
    print("Test 2: API Connections")
    print("="*60)
    
    success = True
    
    # Test Polygon
    try:
        from data_downloader import DataDownloader
        downloader = DataDownloader()
        print("[OK] Polygon.io client initialized")
    except Exception as e:
        print(f"[FAIL] Polygon.io failed: {e}")
        success = False
    
    # Test Sentiment collector
    try:
        from sentiment_collector import SentimentCollector
        collector = SentimentCollector()
        print("[OK] Sentiment collector initialized")
    except Exception as e:
        print(f"[FAIL] Sentiment collector failed: {e}")
        success = False
    
    return success

def test_data_download():
    """Test data download"""
    print("\n" + "="*60)
    print("Test 3: Data Download")
    print("="*60)
    
    try:
        from data_downloader import DataDownloader
        downloader = DataDownloader()
        
        # Try downloading one day of data
        test_date = datetime.now() - timedelta(days=1)
        # Find a weekday
        while test_date.weekday() >= 5:
            test_date -= timedelta(days=1)
        
        print(f"Attempting to download AAPL for {test_date.strftime('%Y-%m-%d')}...")
        df = downloader.download_day('AAPL', test_date)
        
        if df is not None and not df.empty:
            print(f"[OK] Downloaded {len(df)} rows")
            return True
        else:
            print("[FAIL] No data returned")
            return False
    except Exception as e:
        print(f"[FAIL] Download failed: {e}")
        return False

def test_feature_computation():
    """Test feature computation"""
    print("\n" + "="*60)
    print("Test 4: Feature Computation")
    print("="*60)
    
    try:
        import pandas as pd
        import numpy as np
        from technical_indicators import add_all_indicators
        from temporal_features import add_all_temporal_features
        
        # Create sample data
        dates = pd.date_range('2024-01-01 09:30', periods=100, freq='1min')
        df = pd.DataFrame({
            'Open': 100 + np.random.randn(100).cumsum() * 0.1,
            'High': 101 + np.random.randn(100).cumsum() * 0.1,
            'Low': 99 + np.random.randn(100).cumsum() * 0.1,
            'Close': 100 + np.random.randn(100).cumsum() * 0.1,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Add technical indicators
        df = add_all_indicators(df)
        print(f"[OK] Technical indicators: {len([c for c in df.columns if c not in ['Open','High','Low','Close','Volume']])} features")
        
        # Add temporal features
        df = add_all_temporal_features(df, 'TEST')
        print(f"[OK] Temporal features added")
        
        # Drop NaN
        df = df.dropna()
        print(f"[OK] Final data: {len(df)} rows, {len(df.columns)} columns")
        
        return True
    except Exception as e:
        print(f"[FAIL] Feature computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_preparation():
    """Test data preparation"""
    print("\n" + "="*60)
    print("Test 5: Data Preparation")
    print("="*60)
    
    try:
        import pandas as pd
        import numpy as np
        from data_loader import DataLoader
        from technical_indicators import add_all_indicators
        from temporal_features import add_all_temporal_features
        from sentiment_features import add_default_sentiment_features
        
        # Create sample data
        dates = pd.date_range('2024-01-01 09:30', periods=500, freq='1min')
        df = pd.DataFrame({
            'Open': 100 + np.random.randn(500).cumsum() * 0.1,
            'High': 101 + np.random.randn(500).cumsum() * 0.1,
            'Low': 99 + np.random.randn(500).cumsum() * 0.1,
            'Close': 100 + np.random.randn(500).cumsum() * 0.1,
            'Volume': np.random.randint(1000, 10000, 500)
        }, index=dates)
        
        # Add features
        df = add_all_indicators(df)
        df = add_all_temporal_features(df, 'TEST')
        df = add_default_sentiment_features(df)
        df = df.dropna()
        
        # Prepare data
        loader = DataLoader('TEST')
        data = loader.prepare_data_for_training(df)
        
        print(f"[OK] Train set: {data['X_train'].shape}")
        print(f"[OK] Val set:   {data['X_val'].shape}")
        print(f"[OK] Test set:  {data['X_test'].shape}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_building():
    """Test model building"""
    print("\n" + "="*60)
    print("Test 6: Model Building")
    print("="*60)
    
    try:
        from models.lstm_predictor import build_lstm_model
        
        input_shape = (60, 50)  # 60 timesteps, 50 features
        model = build_lstm_model(input_shape)
        
        print(f"[OK] Model built successfully")
        print(f"[OK] Total parameters: {model.count_params():,}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Model building failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SYSTEM TEST - Stock Price Predictor")
    print("="*60)
    
    tests = [
        ("Configuration", test_config),
        ("API Connections", test_api_connections),
        ("Data Download", test_data_download),
        ("Feature Computation", test_feature_computation),
        ("Data Preparation", test_data_preparation),
        ("Model Building", test_model_building),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status:8s} {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n*** All systems operational! ***")
        print("\nYou're ready to:")
        print("  1. Download data: python data_downloader.py --ticker AAPL --days 30")
        print("  2. Process features: python feature_pipeline.py --ticker AAPL --start-date YYYY-MM-DD --end-date YYYY-MM-DD")
        print("  3. Train model: python train_model.py --ticker AAPL --start-date YYYY-MM-DD --end-date YYYY-MM-DD")
    else:
        print("\n*** Some tests failed. Please fix the issues above. ***")
        sys.exit(1)

if __name__ == '__main__':
    main()

