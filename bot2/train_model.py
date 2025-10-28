"""
Train Model - Complete training pipeline

Orchestrates:
1. Loading prepared data
2. Building model
3. Training with callbacks
4. Evaluation on test set
5. Saving model and results
"""
import os
import argparse
from datetime import datetime
import json
from config import MODELS_DIR, validate_config
from utils import setup_logging, Timer
from data_loader import DataLoader
from models.lstm_predictor import build_lstm_model, train_model, evaluate_model
from feature_pipeline import FeaturePipeline

logger = setup_logging(__name__, 'training.log')

def train_pipeline(ticker, start_date, end_date, force_reload=False, 
                   epochs=None, batch_size=None, save_results=True):
    """
    Complete training pipeline
    
    Args:
        ticker: Stock ticker
        start_date: datetime object
        end_date: datetime object
        force_reload: Force reload and reprocess data
        epochs: Number of training epochs
        batch_size: Training batch size
        save_results: Save training results
    
    Returns:
        dict with model, history, and evaluation results
    """
    logger.info(f"Starting training pipeline for {ticker}")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Step 1: Load or prepare data
    data_loader = DataLoader(ticker)
    prepared_data_file = f"{ticker}_prepared_data.pkl"
    
    if not force_reload:
        data = DataLoader.load_prepared_data(ticker, prepared_data_file)
    else:
        data = None
    
    if data is None:
        logger.info("Preparing data from scratch...")
        
        # Load processed features
        pipeline = FeaturePipeline()
        df = pipeline.load_processed_data_range(ticker, start_date, end_date)
        
        if df is None or df.empty:
            logger.error("No processed data found. Run feature_pipeline.py first.")
            return None
        
        # Prepare data for training
        data = data_loader.prepare_data_for_training(df)
        
        # Save for future use
        data_loader.save_prepared_data(data, prepared_data_file)
    
    # Step 2: Build model
    logger.info("Building model...")
    input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])
    model = build_lstm_model(input_shape)
    
    # Print model summary
    logger.info("\nModel architecture:")
    model.summary(print_fn=logger.info)
    logger.info(f"Total parameters: {model.count_params():,}")
    
    # Step 3: Train model
    logger.info("Starting model training...")
    with Timer("Model training"):
        history = train_model(
            model, 
            data, 
            ticker,
            epochs=epochs,
            batch_size=batch_size
        )
    
    # Step 4: Evaluate on test set
    logger.info("Evaluating on test set...")
    results = evaluate_model(model, data)
    
    # Step 5: Save results
    if save_results:
        # Save training history
        history_file = os.path.join(MODELS_DIR, f'{ticker}_training_history.json')
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'classification_accuracy': [float(x) for x in history.history['classification_accuracy']],
            'val_classification_accuracy': [float(x) for x in history.history['val_classification_accuracy']],
        }
        with open(history_file, 'w') as f:
            json.dump(history_dict, f, indent=2)
        logger.info(f"Saved training history to {history_file}")
        
        # Save evaluation results
        results_file = os.path.join(MODELS_DIR, f'{ticker}_evaluation_results.json')
        results_to_save = {
            'ticker': ticker,
            'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'training_date': datetime.now().isoformat(),
            'test_samples': int(len(data['X_test'])),
            'classification_accuracy': float(results['classification_accuracy']),
            'directional_accuracy': float(results['directional_accuracy']),
            'mae': float(results['mae']),
            'rmse': float(results['rmse']),
            'class_accuracies': {k: float(v) for k, v in results['class_accuracies'].items()}
        }
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        logger.info(f"Saved evaluation results to {results_file}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"Training Complete: {ticker}")
    print(f"{'='*60}")
    print(f"Classification Accuracy: {results['classification_accuracy']:.2%}")
    print(f"Directional Accuracy:    {results['directional_accuracy']:.2%}")
    print(f"Regression MAE:          {results['mae']:.4f}")
    print(f"Regression RMSE:         {results['rmse']:.4f}")
    print(f"\nClass-specific accuracies:")
    for name, acc in results['class_accuracies'].items():
        print(f"  {name:8s}: {acc:.2%}")
    print(f"{'='*60}\n")
    
    return {
        'model': model,
        'history': history,
        'results': results,
        'data': data
    }

def main():
    """Command-line interface for training"""
    parser = argparse.ArgumentParser(description='Train stock prediction model')
    parser.add_argument('--ticker', '-t', type=str, required=True,
                       help='Stock ticker symbol')
    parser.add_argument('--start-date', type=str, required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int,
                       help='Training batch size')
    parser.add_argument('--force-reload', action='store_true',
                       help='Force reload and reprocess data')
    
    args = parser.parse_args()
    
    # Validate configuration
    try:
        validate_config()
    except ValueError as e:
        logger.error(str(e))
        return
    
    # Parse dates
    start = datetime.strptime(args.start_date, '%Y-%m-%d')
    end = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Run training pipeline
    result = train_pipeline(
        ticker=args.ticker,
        start_date=start,
        end_date=end,
        force_reload=args.force_reload,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if result:
        logger.info("[SUCCESS] Training pipeline completed successfully!")
    else:
        logger.error("[FAILED] Training pipeline failed")

if __name__ == '__main__':
    main()

