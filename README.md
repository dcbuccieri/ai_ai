# AI Stock Trading Bot

A machine learning project for stock price prediction using historical data.

## Project Structure

```
stock_trading_bot/
├── data/
│   ├── raw/           # Unprocessed datasets
│   ├── processed/     # Cleaned and transformed datasets
│   └── external/      # External data sources
├── src/
│   ├── models/        # ML model definitions
│   ├── preprocessing/ # Data cleaning scripts
│   ├── evaluation/    # Model evaluation scripts
│   └── utils/         # Utility functions
├── notebooks/         # Jupyter notebooks for exploration
├── tests/            # Unit and integration tests
├── configs/          # Configuration files
└── scripts/          # Standalone scripts
```

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run data collection: `python scripts/collect_data.py`
3. Explore data: `jupyter notebook notebooks/`

## Next Steps

- [ ] Set up data collection
- [ ] Implement basic LSTM model
- [ ] Add technical indicators
- [ ] Create evaluation metrics
