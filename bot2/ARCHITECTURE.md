# Architecture Documentation

## System Overview

The stock price predictor is a modular machine learning system that combines price data, technical indicators, and market sentiment to predict stock price movements.

## Core Components

### 1. Data Collection Layer

#### `data_downloader.py`
- **Purpose**: Download OHLCV data from Polygon.io
- **Key Features**:
  - Date-based caching (never re-downloads)
  - Rate limit handling (5 calls/min)
  - Automatic retry logic
  - Weekend detection
- **Storage**: `data/{TICKER}/raw/1m/{YYYY-MM-DD}.pkl`

#### `sentiment_collector.py`
- **Purpose**: Aggregate sentiment from multiple sources
- **Sources**:
  - NewsAPI: Article sentiment analysis
  - Reddit: Social media buzz
  - FRED: Economic indicators (VIX, rates, unemployment)
  - Google Trends: Search interest
- **Output**: Composite sentiment score [-1, 1]

#### `data_manager.py`
- **Purpose**: Track downloaded data and identify gaps
- **Functions**:
  - Metadata tracking
  - Missing date detection
  - Bulk download orchestration

### 2. Feature Engineering Layer

#### `technical_indicators.py`
- **Purpose**: Calculate 25+ technical analysis indicators
- **Indicators**:
  - Momentum: RSI, MACD, ROC
  - Volatility: Bollinger Bands, ATR
  - Trend: ADX, Moving Averages
  - Volume: OBV, VWAP, Volume Ratio
- **Implementation**: Pure Python/NumPy (no TA-Lib required)

#### `temporal_features.py`
- **Purpose**: Extract time-based patterns
- **Features**:
  - Time of day (sin/cos encoding for cyclical)
  - Day of week
  - Market session (opening, mid-day, closing)
  - Days until earnings
  - Market regime detection

#### `sentiment_features.py`
- **Purpose**: Integrate sentiment into feature set
- **Features**:
  - News sentiment and article count
  - Social media sentiment and mentions
  - Economic health composite
  - Market fear index (VIX normalized)

#### `feature_pipeline.py`
- **Purpose**: Orchestrate entire feature engineering process
- **Flow**:
  1. Load raw OHLCV data
  2. Compute technical indicators
  3. Add temporal features
  4. Fetch and add sentiment
  5. Handle NaN values
  6. Save processed data

**Total Features**: 40+ engineered features from 5 raw inputs (OHLCV)

### 3. Data Preparation Layer

#### `data_loader.py`
- **Purpose**: Prepare data for LSTM model
- **Key Functions**:
  - Create target variables (price change %, direction class)
  - Chronological train/val/test split (70/15/15)
  - Feature scaling (RobustScaler)
  - Sliding window generation
  - Sequence creation for LSTM

**Target Variables**:
- **Regression**: Percentage price change (continuous)
- **Classification**: Direction class (DOWN/NEUTRAL/UP)
  - DOWN: < -0.3%
  - NEUTRAL: -0.3% to +0.3%
  - UP: > +0.3%

**Sequence Format**:
- Input: (batch, 60 timesteps, 40+ features)
- Output: Direction probabilities + price change magnitude

### 4. Model Layer

#### `models/lstm_predictor.py`
- **Purpose**: LSTM neural network with attention

**Architecture**:
```
Input (60 timesteps, 40+ features)
    ↓
LSTM Layer 1 (128 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (64 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
Attention Layer (custom)
    ↓
Dense (32 units, ReLU)
    ↓
Dropout (0.2)
    ↓
Split into two heads:
    ├─→ Classification Output (3 units, softmax)
    └─→ Regression Output (1 unit, linear)
```

**Loss Function**:
- Combined loss: 60% classification + 40% regression
- Classification: Categorical crossentropy
- Regression: Mean squared error

**Training**:
- Optimizer: Adam (lr=0.001)
- Callbacks:
  - EarlyStopping (patience=10)
  - ModelCheckpoint (save best)
  - ReduceLROnPlateau (patience=5)
- Typical training time: 5-30 minutes depending on data size

### 5. Prediction Layer

#### `predictor.py`
- **Purpose**: Make predictions with trained model
- **Input**: DataFrame with 60+ timesteps of features
- **Output**:
  - Predicted direction (UP/DOWN/NEUTRAL)
  - Confidence score (0-1)
  - Expected price change (%)
  - Predicted price
  - Should trade flag (confidence >= 65%)

#### `backtester.py`
- **Purpose**: Simulate trading strategy
- **Metrics**:
  - Win rate
  - Profit factor
  - Maximum drawdown
  - Sharpe ratio
  - Alpha (vs buy-and-hold)
- **Transaction Costs**: 0.1% per trade (configurable)

#### `live_predictor.py`
- **Purpose**: Real-time predictions
- **Features**:
  - Automatic data fetching
  - Feature computation
  - Sentiment caching (1 hour TTL)
  - Retry logic
  - Market hours detection
  - Prediction logging

## Data Flow

```
1. Download Data
   data_downloader.py → data/{TICKER}/raw/1m/{DATE}.pkl

2. Collect Sentiment
   sentiment_collector.py → data/{TICKER}/sentiment/{DATE}.pkl

3. Process Features
   feature_pipeline.py:
      Load raw → Add indicators → Add temporal → Add sentiment
      → data/{TICKER}/processed/1m_features/{DATE}.pkl

4. Prepare for Training
   data_loader.py:
      Load processed → Create targets → Split → Scale → Create sequences
      → models/{TICKER}_prepared_data.pkl

5. Train Model
   train_model.py:
      Load prepared data → Build model → Train → Evaluate
      → models/{TICKER}_lstm_best.h5
      → models/{TICKER}_evaluation_results.json

6. Predict
   predictor.py / live_predictor.py:
      Load model → Fetch latest data → Compute features → Predict
      → Prediction with confidence
```

## Configuration

All settings in `config.py`:

### Model Hyperparameters
- `LSTM_UNITS_1 = 128`
- `LSTM_UNITS_2 = 64`
- `DENSE_UNITS = 32`
- `DROPOUT_RATE = 0.2`
- `LEARNING_RATE = 0.001`
- `BATCH_SIZE = 32`
- `EPOCHS = 100`

### Feature Settings
- `LOOKBACK_WINDOW = 60` (minutes of history)
- `PREDICTION_HORIZON = 15` (predict 15 min ahead)
- `DOWN_THRESHOLD = -0.003` (-0.3%)
- `UP_THRESHOLD = 0.003` (+0.3%)

### Trading Settings
- `MIN_CONFIDENCE = 0.65` (65% confidence to trade)
- `TRANSACTION_COST = 0.001` (0.1% per trade)

## Performance Characteristics

### Computational Requirements
- **Training**:
  - CPU: ~10-30 min for 90 days of data
  - GPU: ~2-5 min for 90 days of data
  - Memory: ~2-4 GB
- **Inference**: <100ms per prediction

### Storage Requirements
- Raw data: ~1 MB per day per ticker
- Processed data: ~5 MB per day per ticker
- Model size: ~500 KB - 2 MB

### Expected Accuracy
- **Directional Accuracy**: 55-60% (test set)
- **Classification Accuracy**: 50-65% (all classes)
- **High Confidence Predictions**: 65-75% accuracy
- **Sharpe Ratio**: 1.5-2.5 (backtesting)

## Scalability

### Multiple Tickers
Each ticker is independent:
```bash
for ticker in AAPL TSLA MSFT; do
    python data_downloader.py --ticker $ticker --days 90
    python feature_pipeline.py --ticker $ticker --start ... --end ...
    python train_model.py --ticker $ticker --start ... --end ...
done
```

### Parallel Processing
Feature pipeline can process dates in parallel (future enhancement):
```python
from multiprocessing import Pool
with Pool(4) as p:
    p.map(process_date, date_list)
```

### Database Option
Current: Pickle files (simple, fast)
Future: PostgreSQL/TimescaleDB for:
- Multi-user access
- Better querying
- Time-series optimizations

## Error Handling

### Graceful Degradation
- Missing sentiment → Use default neutral values
- API failure → Retry with exponential backoff
- Missing data → Skip and continue

### Logging
All components log to `logs/` directory:
- `data_downloader.log`
- `sentiment_collector.log`
- `feature_pipeline.log`
- `training.log`
- `live_predictor.log`

### Validation
- Data validation at each step
- Feature count verification
- Model input shape checking
- Prediction sanity checks

## Security

### API Keys
- Stored in `.env` file (gitignored)
- Loaded via python-dotenv
- Never hardcoded

### Data Privacy
- All data stored locally
- No data sent to external servers (except API calls)
- Models trained locally

## Future Enhancements

### Short-term
1. Additional data sources (Twitter, SEC filings)
2. More sophisticated attention mechanism (multi-head)
3. Ensemble models (combine multiple predictions)
4. Hyperparameter optimization (Optuna)

### Long-term
1. Transformer architecture
2. Reinforcement learning for trading strategy
3. Multi-ticker correlation features
4. Real-time news event detection
5. WebSocket data streaming

## Design Principles

1. **Modularity**: Each component is independent and testable
2. **Caching**: Never recompute what's already computed
3. **Fail-Safe**: Graceful degradation on errors
4. **Configurability**: All parameters in config.py
5. **Observability**: Comprehensive logging
6. **Reproducibility**: Fixed random seeds, versioned data

## Testing

```bash
python test_system.py  # Full system test
python -m pytest tests/  # Unit tests (if implemented)
```

## Monitoring

Track model performance over time:
```bash
# Check evaluation metrics
cat models/AAPL_evaluation_results.json

# View prediction history
cat logs/AAPL_predictions.jsonl
```

Retrain weekly or when directional accuracy drops below 52%.

