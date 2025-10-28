"""
Configuration file for stock prediction system
Centralized config to avoid magic numbers throughout codebase
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== API KEYS ====================
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')
FRED_API_KEY = os.getenv('FRED_API_KEY', '')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'stock_predictor_bot_v1.0')

# ==================== DATA SETTINGS ====================
DATA_DIR = 'data'
MODELS_DIR = 'models'
LOGS_DIR = 'logs'

# Rate limits (requests per minute)
POLYGON_RATE_LIMIT = 5  # Free tier
NEWSAPI_RATE_LIMIT = 100  # Per day
REDDIT_RATE_LIMIT = 60  # Per minute

# Data storage structure
RAW_DATA_SUBDIR = 'raw/1m'
PROCESSED_DATA_SUBDIR = 'processed/1m_features'
SENTIMENT_DATA_SUBDIR = 'sentiment'

# ==================== MODEL HYPERPARAMETERS ====================
# LSTM Architecture
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.2

# Training
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Early stopping
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# ==================== FEATURE ENGINEERING ====================
# Lookback window (timesteps to use for prediction)
LOOKBACK_WINDOW = 60  # 60 minutes of 1m data

# Prediction horizon
PREDICTION_HORIZON = 15  # Predict 15 minutes ahead

# Technical Indicator Periods
RSI_PERIOD = 14
RSI_PERIOD_LONG = 28
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14
ADX_PERIOD = 14
STOCH_PERIOD = 14

# ==================== PREDICTION SETTINGS ====================
# Classification thresholds (percentage change)
DOWN_THRESHOLD = -0.003  # -0.3%
UP_THRESHOLD = 0.003     # +0.3%

# Confidence threshold for trading
MIN_CONFIDENCE = 0.65  # Only trade if model confidence >65%

# Transaction costs (for backtesting)
TRANSACTION_COST = 0.001  # 0.1% per trade

# ==================== LOGGING ====================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ==================== MARKET HOURS (US Eastern) ====================
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# ==================== VALIDATION ====================
def validate_config():
    """Validate that critical config values are set"""
    errors = []
    
    if not POLYGON_API_KEY:
        errors.append("POLYGON_API_KEY not set in .env file")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True

if __name__ == '__main__':
    # Test configuration
    validate_config()
    print("✓ Configuration valid")
    print(f"✓ Data directory: {DATA_DIR}")
    print(f"✓ Models directory: {MODELS_DIR}")
    print(f"✓ Polygon API key: {'Set' if POLYGON_API_KEY else 'NOT SET'}")
    print(f"✓ NewsAPI key: {'Set' if NEWSAPI_KEY else 'NOT SET'}")
    print(f"✓ FRED API key: {'Set' if FRED_API_KEY else 'NOT SET'}")

