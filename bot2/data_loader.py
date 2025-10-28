"""
Data Loader - Prepare data for model training

Handles:
- Loading processed feature data
- Creating train/val/test splits (chronological)
- Generating sliding windows for LSTM
- Creating target variables (price change, direction)
- Feature scaling
- Batch generation
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from config import (
    LOOKBACK_WINDOW, PREDICTION_HORIZON, DOWN_THRESHOLD, UP_THRESHOLD,
    VALIDATION_SPLIT, TEST_SPLIT, MODELS_DIR
)
from utils import setup_logging, save_pickle, load_pickle, percentage_change

logger = setup_logging(__name__, 'data_loader.log')

class DataLoader:
    """Load and prepare data for model training"""
    
    def __init__(self, ticker):
        """
        Initialize data loader
        
        Args:
            ticker: Stock ticker symbol
        """
        self.ticker = ticker
        self.scaler = RobustScaler()  # Robust to outliers
        self.feature_columns = None
        self.target_columns = ['price_change', 'direction_class']
    
    def create_target_variables(self, df):
        """
        Create target variables for prediction
        
        Args:
            df: DataFrame with price data
        
        Returns:
            DataFrame with target variables added
        """
        df = df.copy()
        
        # Calculate future price change (percentage)
        future_price = df['Close'].shift(-PREDICTION_HORIZON)
        df['price_change'] = percentage_change(future_price, df['Close'])
        
        # Create classification labels
        # 0 = DOWN, 1 = NEUTRAL, 2 = UP
        df['direction_class'] = 1  # Default to NEUTRAL
        df.loc[df['price_change'] < DOWN_THRESHOLD, 'direction_class'] = 0  # DOWN
        df.loc[df['price_change'] > UP_THRESHOLD, 'direction_class'] = 2  # UP
        
        # Drop last PREDICTION_HORIZON rows (no future data available)
        df = df.iloc[:-PREDICTION_HORIZON]
        
        logger.info(f"Created target variables. Class distribution:")
        logger.info(f"  DOWN (0): {(df['direction_class'] == 0).sum()}")
        logger.info(f"  NEUTRAL (1): {(df['direction_class'] == 1).sum()}")
        logger.info(f"  UP (2): {(df['direction_class'] == 2).sum()}")
        
        return df
    
    def split_data(self, df, validation_split=None, test_split=None):
        """
        Split data chronologically into train/val/test
        
        Args:
            df: DataFrame with features and targets
            validation_split: Fraction for validation (default from config)
            test_split: Fraction for test (default from config)
        
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        val_split = validation_split or VALIDATION_SPLIT
        test_split = test_split or TEST_SPLIT
        
        n = len(df)
        
        # Calculate split indices (chronological)
        train_end = int(n * (1 - val_split - test_split))
        val_end = int(n * (1 - test_split))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(f"Data split:")
        logger.info(f"  Train: {len(train_df)} rows ({len(train_df)/n*100:.1f}%)")
        logger.info(f"  Val:   {len(val_df)} rows ({len(val_df)/n*100:.1f}%)")
        logger.info(f"  Test:  {len(test_df)} rows ({len(test_df)/n*100:.1f}%)")
        logger.info(f"  Train date range: {train_df.index.min()} to {train_df.index.max()}")
        logger.info(f"  Val date range:   {val_df.index.min()} to {val_df.index.max()}")
        logger.info(f"  Test date range:  {test_df.index.min()} to {test_df.index.max()}")
        
        return train_df, val_df, test_df
    
    def fit_scaler(self, train_df):
        """
        Fit scaler on training data
        
        Args:
            train_df: Training DataFrame
        """
        # Identify feature columns (exclude OHLCV and targets)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] + self.target_columns
        self.feature_columns = [col for col in train_df.columns if col not in exclude_cols]
        
        # Fit scaler on training features only
        self.scaler.fit(train_df[self.feature_columns])
        
        logger.info(f"Fitted scaler on {len(self.feature_columns)} features")
    
    def scale_features(self, df):
        """
        Scale features using fitted scaler
        
        Args:
            df: DataFrame to scale
        
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        return df
    
    def create_sequences(self, df, lookback=None):
        """
        Create sliding window sequences for LSTM
        
        Args:
            df: DataFrame with features and targets
            lookback: Lookback window size (default from config)
        
        Returns:
            tuple: (X, y_regression, y_classification)
                X: shape (samples, timesteps, features)
                y_regression: shape (samples,) - price change
                y_classification: shape (samples, 3) - one-hot encoded direction
        """
        lookback = lookback or LOOKBACK_WINDOW
        
        if self.feature_columns is None:
            raise ValueError("Must call fit_scaler first")
        
        # Extract features and targets
        features = df[self.feature_columns].values
        price_changes = df['price_change'].values
        direction_classes = df['direction_class'].values
        
        X = []
        y_reg = []
        y_class = []
        
        # Create sequences
        for i in range(lookback, len(df)):
            # Features: last 'lookback' timesteps
            X.append(features[i-lookback:i])
            
            # Targets: current timestep (already shifted to future in create_target_variables)
            y_reg.append(price_changes[i-1])
            y_class.append(direction_classes[i-1])
        
        X = np.array(X)
        y_reg = np.array(y_reg)
        y_class = np.array(y_class)
        
        # One-hot encode classification targets
        y_class_onehot = np.zeros((len(y_class), 3))
        y_class_onehot[np.arange(len(y_class)), y_class.astype(int)] = 1
        
        logger.info(f"Created sequences:")
        logger.info(f"  X shape: {X.shape} (samples, timesteps, features)")
        logger.info(f"  y_regression shape: {y_reg.shape}")
        logger.info(f"  y_classification shape: {y_class_onehot.shape}")
        
        return X, y_reg, y_class_onehot
    
    def prepare_data_for_training(self, df):
        """
        Complete data preparation pipeline
        
        Args:
            df: Raw DataFrame with all features
        
        Returns:
            dict with train/val/test data ready for model
        """
        logger.info("Starting data preparation pipeline...")
        
        # Create target variables
        df = self.create_target_variables(df)
        
        # Split data chronologically
        train_df, val_df, test_df = self.split_data(df)
        
        # Fit scaler on training data
        self.fit_scaler(train_df)
        
        # Scale all splits
        train_df = self.scale_features(train_df)
        val_df = self.scale_features(val_df)
        test_df = self.scale_features(test_df)
        
        # Create sequences
        X_train, y_train_reg, y_train_class = self.create_sequences(train_df)
        X_val, y_val_reg, y_val_class = self.create_sequences(val_df)
        X_test, y_test_reg, y_test_class = self.create_sequences(test_df)
        
        # Package everything
        data = {
            'X_train': X_train,
            'y_train_reg': y_train_reg,
            'y_train_class': y_train_class,
            'X_val': X_val,
            'y_val_reg': y_val_reg,
            'y_val_class': y_val_class,
            'X_test': X_test,
            'y_test_reg': y_test_reg,
            'y_test_class': y_test_class,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'ticker': self.ticker
        }
        
        logger.info("Data preparation complete!")
        
        return data
    
    def save_prepared_data(self, data, filename=None):
        """
        Save prepared data to disk
        
        Args:
            data: dict from prepare_data_for_training
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"{self.ticker}_prepared_data.pkl"
        
        filepath = os.path.join(MODELS_DIR, filename)
        save_pickle(data, filepath)
        logger.info(f"Saved prepared data to {filepath}")
    
    @staticmethod
    def load_prepared_data(ticker, filename=None):
        """
        Load prepared data from disk
        
        Args:
            ticker: Stock ticker
            filename: Optional custom filename
        
        Returns:
            dict with prepared data
        """
        if filename is None:
            filename = f"{ticker}_prepared_data.pkl"
        
        filepath = os.path.join(MODELS_DIR, filename)
        data = load_pickle(filepath)
        
        if data:
            logger.info(f"Loaded prepared data from {filepath}")
        
        return data

def main():
    """Test data loader"""
    import argparse
    from datetime import datetime
    from feature_pipeline import FeaturePipeline
    
    parser = argparse.ArgumentParser(description='Prepare data for training')
    parser.add_argument('--ticker', '-t', type=str, required=True)
    parser.add_argument('--start-date', type=str, required=True)
    parser.add_argument('--end-date', type=str, required=True)
    parser.add_argument('--save', action='store_true', help='Save prepared data')
    
    args = parser.parse_args()
    
    start = datetime.strptime(args.start_date, '%Y-%m-%d')
    end = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Load processed data
    pipeline = FeaturePipeline()
    df = pipeline.load_processed_data_range(args.ticker, start, end)
    
    if df is None:
        print("No data found. Run feature_pipeline.py first.")
        return
    
    # Prepare data
    loader = DataLoader(args.ticker)
    data = loader.prepare_data_for_training(df)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Data Preparation Summary: {args.ticker}")
    print(f"{'='*60}")
    print(f"Features: {len(data['feature_columns'])}")
    print(f"\nTraining set:")
    print(f"  X shape: {data['X_train'].shape}")
    print(f"  y_reg shape: {data['y_train_reg'].shape}")
    print(f"  y_class shape: {data['y_train_class'].shape}")
    print(f"\nValidation set:")
    print(f"  X shape: {data['X_val'].shape}")
    print(f"\nTest set:")
    print(f"  X shape: {data['X_test'].shape}")
    print(f"{'='*60}\n")
    
    # Save if requested
    if args.save:
        loader.save_prepared_data(data)
        print("✓ Data saved!")

if __name__ == '__main__':
    main()

