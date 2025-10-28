# Quick Start Guide - Stock Price Predictor

Get your stock prediction system up and running in minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 2GB+ free disk space for data
- API keys (free tier is fine)

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note on TA-Lib**: If installation fails, the system will work without it using pure Python implementations.

## Step 2: Configure API Keys

1. Copy the template:
   ```bash
   # .env file is created, edit it with your keys
   ```

2. Add your FREE API keys to `.env`:
   ```
   POLYGON_API_KEY=your_key_here
   NEWSAPI_KEY=your_key_here
   FRED_API_KEY=your_key_here
   ```

3. Get free API keys:
   - **Polygon.io**: https://polygon.io/ (5 calls/min free)
   - **NewsAPI**: https://newsapi.org/ (100/day free)
   - **FRED**: https://fred.stlouisfed.org/docs/api/ (unlimited free)

## Step 3: Test Your Setup

```bash
python test_system.py
```

This will verify all components are working. Should see `All systems operational!`

## Step 4: Download Training Data

Start with a small dataset (1 week for testing):

```bash
python data_downloader.py --ticker AAPL --days 7
```

For full training, download more data (30-90 days recommended):

```bash
python data_downloader.py --ticker AAPL --days 90
```

**Tip**: The free Polygon tier has 5 calls/min limit. Downloading 90 days takes ~20 minutes.

## Step 5: Process Features

```bash
python feature_pipeline.py --ticker AAPL --start-date 2024-11-01 --end-date 2024-11-28
```

Replace dates with your downloaded data range. This computes 40+ technical and sentiment features.

## Step 6: Train the Model

```bash
python train_model.py --ticker AAPL --start-date 2024-11-01 --end-date 2024-11-28
```

Training typically takes:
- 1 week of data: ~2 minutes
- 1 month of data: ~5-10 minutes
- 3 months of data: ~15-30 minutes

## Step 7: Make Predictions

### Single Prediction

```bash
python predictor.py --ticker AAPL
```

This uses the most recent data and outputs:
```
==================================================
Prediction for AAPL
==================================================
Current Price:       $178.50
Predicted Direction: UP
Predicted Change:    +0.32%
Predicted Price:     $179.07
Confidence:          73.5%

Class Probabilities:
  DOWN    : 12.3%
  NEUTRAL : 14.2%
  UP      : 73.5%

Trade Signal: ✓ TRADE
              (confidence ≥ 65%)
==================================================
```

### Live Predictions

```bash
python live_predictor.py --ticker AAPL --interval 5
```

Makes predictions every 5 minutes during market hours.

## Common Issues & Solutions

### Issue: "Polygon API key not set"
**Solution**: Add your Polygon API key to `.env` file

### Issue: "No data available"
**Solution**: Run `data_downloader.py` first to download data

### Issue: "Model not found"
**Solution**: Train a model first with `train_model.py`

### Issue: "Insufficient data for prediction"
**Solution**: Need at least 60 minutes of data (LOOKBACK_WINDOW)

### Issue: TA-Lib installation fails
**Solution**: The system works without it using pure Python implementations

## What's Next?

### Improve Model Performance

1. **More Data**: Download 6-12 months for better patterns
   ```bash
   python data_downloader.py --ticker AAPL --days 365
   ```

2. **Multiple Tickers**: Train separate models for different stocks
   ```bash
   python data_downloader.py --ticker TSLA --days 90
   python feature_pipeline.py --ticker TSLA --start-date ... --end-date ...
   python train_model.py --ticker TSLA --start-date ... --end-date ...
   ```

3. **Hyperparameter Tuning**: Edit `config.py` to adjust:
   - `LSTM_UNITS_1`, `LSTM_UNITS_2` (network size)
   - `LOOKBACK_WINDOW` (how many minutes of history)
   - `PREDICTION_HORIZON` (predict how far ahead)
   - `DROPOUT_RATE` (regularization strength)

### Monitor Performance

Check evaluation results:
```bash
cat models/AAPL_evaluation_results.json
```

Review training history:
```bash
cat models/AAPL_training_history.json
```

### Backtest Strategy

Create custom backtest script using `backtester.py` module to simulate trading with your model.

## Project Structure

```
bot2/
├── data/                    # Downloaded and processed data
│   └── {TICKER}/
│       ├── raw/1m/         # Raw OHLCV data
│       ├── processed/      # Features
│       └── sentiment/      # Sentiment data
├── models/                 # Trained models
│   ├── {TICKER}_lstm_best.h5
│   └── {TICKER}_prepared_data.pkl
├── logs/                   # Application logs
├── config.py              # Configuration
├── data_downloader.py     # Download data
├── sentiment_collector.py # Collect sentiment
├── feature_pipeline.py    # Feature engineering
├── train_model.py         # Train model
├── predictor.py           # Make predictions
└── live_predictor.py      # Real-time predictions
```

## Tips for Best Results

1. **Data Quality**: More data = better predictions (but diminishing returns after ~6 months)

2. **Market Conditions**: Model trained in bull market may underperform in bear market. Retrain monthly.

3. **Confidence Threshold**: Only trade when confidence >65% (configurable in `config.py`)

4. **Transaction Costs**: Account for broker fees. Set `TRANSACTION_COST` in `config.py`

5. **Paper Trading**: Test with paper trading before risking real money

6. **Risk Management**: Never risk more than 1-2% of capital on a single trade

## Need Help?

- Check logs in `logs/` directory for detailed error messages
- Run `python test_system.py` to diagnose issues
- Review `.cursor/project_checklist.md` for implementation details

## Disclaimer

This software is for educational purposes only. Past performance does not guarantee future results. Always perform thorough testing before live trading.

