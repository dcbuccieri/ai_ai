"""
Predictor - Make predictions using trained model

Handles:
- Loading trained model and scaler
- Preprocessing new data
- Making predictions with confidence scores
- Output formatting
"""
import os
import numpy as np
from tensorflow import keras
from config import MODELS_DIR, LOOKBACK_WINDOW, MIN_CONFIDENCE
from utils import setup_logging, load_pickle, format_percentage

logger = setup_logging(__name__)

class StockPredictor:
    """Make predictions using trained model"""
    
    def __init__(self, ticker):
        """
        Initialize predictor
        
        Args:
            ticker: Stock ticker symbol
        """
        self.ticker = ticker
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.loaded = False
    
    def load_model(self, model_path=None):
        """
        Load trained model
        
        Args:
            model_path: Optional custom model path
        """
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, f'{self.ticker}_lstm_best.h5')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model with custom objects (AttentionLayer)
        from models.lstm_predictor import AttentionLayer
        self.model = keras.models.load_model(
            model_path,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        
        logger.info(f"Loaded model from {model_path}")
    
    def load_scaler(self, scaler_path=None):
        """
        Load fitted scaler and feature columns
        
        Args:
            scaler_path: Optional custom scaler path
        """
        if scaler_path is None:
            scaler_path = os.path.join(MODELS_DIR, f'{self.ticker}_prepared_data.pkl')
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Prepared data not found: {scaler_path}")
        
        # Load prepared data (contains scaler and feature columns)
        data = load_pickle(scaler_path)
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        
        logger.info(f"Loaded scaler with {len(self.feature_columns)} features")
    
    def initialize(self):
        """Load both model and scaler"""
        self.load_model()
        self.load_scaler()
        self.loaded = True
        logger.info("Predictor initialized and ready")
    
    def preprocess_data(self, df):
        """
        Preprocess data for prediction
        
        Args:
            df: DataFrame with features
        
        Returns:
            numpy array ready for model input
        """
        if not self.loaded:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")
        
        # Ensure we have all required features
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Extract and scale features
        features = df[self.feature_columns].values
        features_scaled = self.scaler.transform(features)
        
        # Create sequence (need at least LOOKBACK_WINDOW rows)
        if len(features_scaled) < LOOKBACK_WINDOW:
            raise ValueError(f"Need at least {LOOKBACK_WINDOW} rows, got {len(features_scaled)}")
        
        # Take last LOOKBACK_WINDOW rows
        sequence = features_scaled[-LOOKBACK_WINDOW:]
        
        # Reshape for model: (1, timesteps, features)
        X = sequence.reshape(1, LOOKBACK_WINDOW, len(self.feature_columns))
        
        return X
    
    def predict(self, df, return_confidence=True):
        """
        Make prediction
        
        Args:
            df: DataFrame with features (must have at least LOOKBACK_WINDOW rows)
            return_confidence: If True, return confidence score
        
        Returns:
            dict with prediction details
        """
        if not self.loaded:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")
        
        # Preprocess
        X = self.preprocess_data(df)
        
        # Predict
        y_class_probs, y_reg = self.model.predict(X, verbose=0)
        
        # Extract predictions
        class_probs = y_class_probs[0]  # [prob_down, prob_neutral, prob_up]
        price_change = float(y_reg[0][0])
        
        # Determine predicted class and confidence
        predicted_class_idx = np.argmax(class_probs)
        confidence = float(class_probs[predicted_class_idx])
        
        class_names = ['DOWN', 'NEUTRAL', 'UP']
        predicted_direction = class_names[predicted_class_idx]
        
        # Get current price for target calculation
        current_price = float(df['Close'].iloc[-1])
        predicted_price = current_price * (1 + price_change)
        
        prediction = {
            'ticker': self.ticker,
            'current_price': current_price,
            'predicted_direction': predicted_direction,
            'predicted_change_pct': price_change,
            'predicted_price': predicted_price,
            'confidence': confidence,
            'class_probabilities': {
                'DOWN': float(class_probs[0]),
                'NEUTRAL': float(class_probs[1]),
                'UP': float(class_probs[2])
            },
            'should_trade': confidence >= MIN_CONFIDENCE
        }
        
        return prediction
    
    def format_prediction(self, prediction):
        """
        Format prediction for display
        
        Args:
            prediction: dict from predict()
        
        Returns:
            Formatted string
        """
        lines = []
        lines.append(f"\n{'='*50}")
        lines.append(f"Prediction for {prediction['ticker']}")
        lines.append(f"{'='*50}")
        lines.append(f"Current Price:       ${prediction['current_price']:.2f}")
        lines.append(f"Predicted Direction: {prediction['predicted_direction']}")
        lines.append(f"Predicted Change:    {format_percentage(prediction['predicted_change_pct'])}")
        lines.append(f"Predicted Price:     ${prediction['predicted_price']:.2f}")
        lines.append(f"Confidence:          {prediction['confidence']:.1%}")
        lines.append(f"\nClass Probabilities:")
        for cls, prob in prediction['class_probabilities'].items():
            lines.append(f"  {cls:8s}: {prob:.1%}")
        lines.append(f"\nTrade Signal: {'[TRADE]' if prediction['should_trade'] else '[DO NOT TRADE]'}")
        lines.append(f"              (confidence {'>=' if prediction['should_trade'] else '<'} {MIN_CONFIDENCE:.0%})")
        lines.append(f"{'='*50}\n")
        
        return '\n'.join(lines)

def main():
    """Command-line interface for predictor"""
    import argparse
    from datetime import datetime
    from feature_pipeline import FeaturePipeline
    
    parser = argparse.ArgumentParser(description='Make stock price predictions')
    parser.add_argument('--ticker', '-t', type=str, required=True,
                       help='Stock ticker symbol')
    parser.add_argument('--date', '-d', type=str,
                       help='Date to predict from (YYYY-MM-DD), defaults to latest available')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = StockPredictor(args.ticker)
    
    try:
        predictor.initialize()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure you've trained a model for {args.ticker} first.")
        return
    
    # Load data
    if args.date:
        date = datetime.strptime(args.date, '%Y-%m-%d')
    else:
        # Use latest available processed data
        import os
        from config import DATA_DIR, PROCESSED_DATA_SUBDIR
        
        processed_dir = os.path.join(DATA_DIR, args.ticker, PROCESSED_DATA_SUBDIR)
        if not os.path.exists(processed_dir):
            print(f"No processed data found for {args.ticker}")
            return
        
        files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.pkl')])
        if not files:
            print(f"No processed data files for {args.ticker}")
            return
        
        date_str = files[-1].replace('.pkl', '')
        date = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Load processed data for the date
    pipeline = FeaturePipeline()
    df = pipeline.process_day(args.ticker, date)
    
    if df is None or df.empty:
        print(f"No data available for {args.ticker} on {date.strftime('%Y-%m-%d')}")
        return
    
    # Make prediction
    prediction = predictor.predict(df)
    
    # Display result
    print(predictor.format_prediction(prediction))

if __name__ == '__main__':
    main()

