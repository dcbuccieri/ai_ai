# Stock Price Predictor - Implementation Checklist

## Phase 1: Project Setup ✓
- [x] Create fresh project directory structure
- [x] Initialize git repo with `.gitignore`
- [x] Create `.env.template` with placeholder API keys
- [x] Create actual `.env` for real keys
- [x] Create requirements.txt with all dependencies
- [ ] Install dependencies
- [ ] Test API connections with simple scripts

## Phase 2: Data Collection Infrastructure
- [ ] Build `data_downloader.py`: Polygon.io client for OHLCV data
  - Implement date-based caching (skip existing files)
  - Handle rate limits (5 calls/min) with sleep
  - Store as `data/{TICKER}/raw/1m/{YYYY-MM-DD}.pkl`
  - Add logging for debugging
- [ ] Build `sentiment_collector.py`: Multi-source sentiment aggregator
  - NewsAPI client with daily caching
  - Reddit API with rate limit handling
  - FRED economic indicators (daily updates sufficient)
  - Google Trends (weekly updates)
- [ ] Build `data_manager.py`: Metadata tracking
  - JSON file tracking downloaded dates
  - Function to identify missing data gaps
  - Bulk download orchestrator
- [ ] Download initial dataset: Start with 1 week of AAPL 1m data for testing

## Phase 3: Feature Engineering Pipeline
- [ ] Build `technical_indicators.py`: TA implementations
  - RSI (14, 28 period)
  - MACD (12, 26, 9)
  - Bollinger Bands (20, 2σ)
  - ATR, ADX, Stochastic
  - Volume indicators: OBV, VWAP
  - Support/resistance detection
- [ ] Build `temporal_features.py`:
  - Time of day encoding (sin/cos for cyclical)
  - Day of week one-hot encoding
  - Days until earnings (hardcode quarterly estimates initially)
  - Market session (pre-market, open, mid-day, close)
- [ ] Build `sentiment_features.py`:
  - Aggregate news sentiment score
  - Social media buzz metrics
  - VIX normalization
  - Composite sentiment index
- [ ] Build `feature_pipeline.py`: Main orchestrator
  - Load raw data → compute all features → save to processed/
  - Handle NaN values (forward fill, drop initial rows)
  - Feature normalization/scaling (fit on train, apply to all)
  - Create sliding windows for LSTM input

## Phase 4: Data Preprocessing & Splits
- [ ] Build `data_loader.py`:
  - Load processed feature files for date range
  - Create chronological train/val/test split (70/15/15)
  - Generate sliding windows (e.g., 60 timesteps → predict next 15min)
  - Batch generator for memory efficiency
- [ ] Build target variable:
  - Calculate percentage change 15 minutes ahead
  - Create classification labels: DOWN (<-0.3%), NEUTRAL, UP (>0.3%)
  - Store both regression target and class label
- [ ] Data validation:
  - Check for lookahead bias (no future data in features)
  - Verify no data leakage between splits
  - Inspect feature distributions

## Phase 5: Model Architecture
- [ ] Build `models/lstm_predictor.py`:
  - Input shape: (batch, timesteps, features)
  - LSTM layer 1: 128 units, return_sequences=True
  - Dropout: 0.2
  - LSTM layer 2: 64 units, return_sequences=True
  - Attention layer (custom or use keras)
  - Dense: 32 units, ReLU
  - Output heads:
    - Classification: 3 units, softmax (down/neutral/up)
    - Regression: 1 unit, linear (percentage change)
- [ ] Build custom loss function:
  - Combined classification (categorical crossentropy) + regression (MSE)
  - Weighted: 60% classification, 40% regression
- [ ] Compile model:
  - Optimizer: Adam (lr=0.001)
  - Metrics: Accuracy, MAE, custom profit metric
- [ ] Implement callbacks:
  - EarlyStopping (patience=10, monitor val_loss)
  - ModelCheckpoint (save best model)
  - ReduceLROnPlateau (factor=0.5, patience=5)
  - TensorBoard logging

## Phase 6: Training & Validation
- [ ] Initial training run (small dataset):
  - 1 week of data, verify pipeline works end-to-end
  - Check for errors, memory issues
  - Inspect predictions on validation set
- [ ] Full training:
  - Train on 2 years of data
  - Epochs: 50-100 (early stopping will halt)
  - Batch size: 32 or 64 (tune based on GPU memory)
  - Monitor training/validation loss curves (check overfitting)
- [ ] Hyperparameter tuning:
  - Vary LSTM units, dropout rate, learning rate
  - Try different lookback windows (30, 60, 120 timesteps)
  - Adjust prediction horizon (5min, 15min, 30min)
- [ ] Model evaluation:
  - Directional accuracy on test set
  - Confusion matrix
  - Profit simulation (assuming 0.1% transaction costs)
  - Sharpe ratio calculation

## Phase 7: Prediction & Inference
- [ ] Build `predictor.py`:
  - Load trained model and scalers
  - Accept current market data + features
  - Output prediction with probability
  - Confidence thresholding (only predict if prob >65%)
- [ ] Build `backtester.py`:
  - Walk-forward simulation on test set
  - Implement entry/exit logic based on predictions
  - Track: win rate, profit factor, max drawdown
  - Compare to buy-and-hold baseline
- [ ] Visualization (optional but useful):
  - Plot predictions vs actual prices
  - Show confidence intervals
  - Highlight profitable vs losing trades

## Phase 8: Production Readiness
- [ ] Build `live_predictor.py`:
  - Fetch latest 1m data from Polygon
  - Compute features in real-time
  - Make prediction every minute
  - Log predictions to file
- [ ] Error handling:
  - Retry logic for API failures
  - Fallback to cached data if API down
  - Alert system for model anomalies (predictions outside expected range)
- [ ] Performance optimization:
  - Profile feature computation bottlenecks
  - Cache sentiment data (update hourly, not every minute)
  - Use model inference optimization (TensorFlow Lite if needed)
- [ ] Testing:
  - Unit tests for feature calculations
  - Integration test for full pipeline
  - Stress test with market data edge cases (halts, extreme volatility)

## Phase 9: Monitoring & Iteration (Ongoing)
- [ ] Daily tasks:
  - Download previous day's data
  - Retrain model weekly with new data
  - Monitor prediction accuracy
  - Track profitability metrics
- [ ] Monthly improvements:
  - Add new features (if identified as valuable)
  - Experiment with model architectures
  - Analyze failure modes (when does model perform poorly?)
  - Update sentiment sources if APIs change

## Phase 10: Documentation
- [ ] Create `README.md` with:
  - Setup instructions
  - API key configuration
  - How to download data
  - How to train model
  - How to make predictions
- [ ] Create `ARCHITECTURE.md`: System design overview
- [ ] Create `RESULTS.md`: Model performance metrics
- [ ] Code comments and docstrings

---

## Success Metrics
- **Directional accuracy**: >55% on test set
- **Sharpe ratio**: >1.5 in backtesting
- **Max drawdown**: <15%
- **Training time**: <4 hours on consumer GPU
- **Inference latency**: <100ms per prediction

