"""
Backtester - Simulate trading strategy using model predictions

Evaluates:
- Win rate
- Profit factor
- Maximum drawdown
- Sharpe ratio
- Comparison to buy-and-hold
"""
import numpy as np
import pandas as pd
from config import MIN_CONFIDENCE, TRANSACTION_COST
from utils import setup_logging, format_money, format_percentage

logger = setup_logging(__name__)

class Backtester:
    """Backtest trading strategy on historical data"""
    
    def __init__(self, initial_capital=10000):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital in dollars
        """
        self.initial_capital = initial_capital
        self.reset()
    
    def reset(self):
        """Reset backtester state"""
        self.trades = []
        self.equity_curve = []
        self.capital = self.initial_capital
        self.position = None  # None, 'LONG', or 'SHORT'
        self.position_size = 0
        self.entry_price = 0
    
    def execute_trade(self, timestamp, price, action, confidence):
        """
        Execute a trade
        
        Args:
            timestamp: Trade timestamp
            price: Current price
            action: 'BUY', 'SELL', or 'HOLD'
            confidence: Model confidence (0-1)
        """
        trade_record = {
            'timestamp': timestamp,
            'action': action,
            'price': price,
            'confidence': confidence,
            'capital_before': self.capital,
            'position_before': self.position
        }
        
        # Close existing position if any
        if self.position is not None and action != 'HOLD':
            # Calculate P&L
            if self.position == 'LONG':
                pnl = (price - self.entry_price) * self.position_size
            else:  # SHORT
                pnl = (self.entry_price - price) * self.position_size
            
            # Apply transaction cost
            pnl -= self.entry_price * self.position_size * TRANSACTION_COST
            pnl -= price * self.position_size * TRANSACTION_COST
            
            self.capital += pnl
            
            trade_record['pnl'] = pnl
            trade_record['return'] = pnl / (self.entry_price * self.position_size)
            
            self.position = None
            self.position_size = 0
        
        # Open new position
        if action == 'BUY':
            # Go long
            self.position = 'LONG'
            self.position_size = self.capital / price
            self.entry_price = price
        
        elif action == 'SELL':
            # Go short (if supported)
            # For simplicity, we'll just close long or skip
            pass
        
        trade_record['capital_after'] = self.capital
        trade_record['position_after'] = self.position
        
        self.trades.append(trade_record)
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': self.capital
        })
    
    def run_backtest(self, predictions_df, prices_df):
        """
        Run backtest on predictions
        
        Args:
            predictions_df: DataFrame with predictions (direction, confidence)
            prices_df: DataFrame with actual prices
        
        Returns:
            dict with backtest results
        """
        self.reset()
        
        logger.info(f"Running backtest on {len(predictions_df)} predictions")
        
        for idx in range(len(predictions_df)):
            timestamp = predictions_df.index[idx]
            prediction = predictions_df.iloc[idx]
            
            # Get actual price
            if timestamp not in prices_df.index:
                continue
            
            price = prices_df.loc[timestamp, 'Close']
            
            # Trading logic
            direction = prediction['predicted_direction']
            confidence = prediction['confidence']
            
            # Only trade if confidence exceeds threshold
            if confidence < MIN_CONFIDENCE:
                action = 'HOLD'
            elif direction == 'UP':
                action = 'BUY'
            elif direction == 'DOWN':
                action = 'SELL'  # or 'HOLD' if no shorting
            else:
                action = 'HOLD'
            
            self.execute_trade(timestamp, price, action, confidence)
        
        # Close any remaining position
        if self.position is not None and len(prices_df) > 0:
            final_price = prices_df.iloc[-1]['Close']
            self.execute_trade(prices_df.index[-1], final_price, 'HOLD', 0)
        
        # Calculate metrics
        results = self.calculate_metrics(prices_df)
        
        return results
    
    def calculate_metrics(self, prices_df):
        """
        Calculate backtest metrics
        
        Args:
            prices_df: DataFrame with prices
        
        Returns:
            dict with metrics
        """
        # Basic metrics
        final_capital = self.capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # Trade metrics
        completed_trades = [t for t in self.trades if 'pnl' in t]
        n_trades = len(completed_trades)
        
        if n_trades == 0:
            return {
                'total_return': total_return,
                'final_capital': final_capital,
                'n_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'buy_hold_return': 0
            }
        
        # Win/loss metrics
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / n_trades if n_trades > 0 else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
        
        total_wins = sum([t['pnl'] for t in winning_trades])
        total_losses = abs(sum([t['pnl'] for t in losing_trades]))
        
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Drawdown
        equity_curve = pd.DataFrame(self.equity_curve)
        if not equity_curve.empty:
            equity_curve['cummax'] = equity_curve['equity'].cummax()
            equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['cummax']) / equity_curve['cummax']
            max_drawdown = abs(equity_curve['drawdown'].min())
        else:
            max_drawdown = 0
        
        # Sharpe ratio (simplified - using trade returns)
        if len(completed_trades) > 1:
            returns = [t['return'] for t in completed_trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Buy and hold comparison
        if len(prices_df) > 0:
            initial_price = prices_df.iloc[0]['Close']
            final_price = prices_df.iloc[-1]['Close']
            buy_hold_return = (final_price - initial_price) / initial_price
        else:
            buy_hold_return = 0
        
        results = {
            'total_return': total_return,
            'final_capital': final_capital,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'buy_hold_return': buy_hold_return,
            'alpha': total_return - buy_hold_return  # Excess return vs buy-hold
        }
        
        return results
    
    def print_results(self, results):
        """
        Print backtest results
        
        Args:
            results: dict from calculate_metrics
        """
        print(f"\n{'='*60}")
        print(f"Backtest Results")
        print(f"{'='*60}")
        print(f"Initial Capital:     {format_money(self.initial_capital)}")
        print(f"Final Capital:       {format_money(results['final_capital'])}")
        print(f"Total Return:        {format_percentage(results['total_return'])}")
        print(f"\nTrading Statistics:")
        print(f"  Total Trades:      {results['n_trades']}")
        print(f"  Win Rate:          {format_percentage(results['win_rate'])}")
        print(f"  Average Win:       {format_money(results['avg_win'])}")
        print(f"  Average Loss:      {format_money(results['avg_loss'])}")
        print(f"  Profit Factor:     {results['profit_factor']:.2f}")
        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown:      {format_percentage(results['max_drawdown'])}")
        print(f"  Sharpe Ratio:      {results['sharpe_ratio']:.2f}")
        print(f"\nComparison:")
        print(f"  Buy & Hold Return: {format_percentage(results['buy_hold_return'])}")
        print(f"  Alpha (excess):    {format_percentage(results['alpha'])}")
        print(f"{'='*60}\n")

def main():
    """Test backtester with sample data"""
    # This is a placeholder - in reality, you'd load actual predictions and prices
    print("Backtester module loaded successfully")
    print("Use this class in conjunction with trained model predictions")

if __name__ == '__main__':
    main()

