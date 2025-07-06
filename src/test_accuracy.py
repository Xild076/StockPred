import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from colorama import Fore, Style

try:
    from .modeling.engine import UniversalModelEngine
    from . import config
except (ImportError, ValueError):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.modeling.engine import UniversalModelEngine
    from src import config

last_results = None

class AccuracyTester:
    def __init__(self):
        self.data = pd.read_parquet(config.DATA_PATH)
        self.engine = UniversalModelEngine(self.data)
        self.load_trained_model()
        
    def load_trained_model(self):
        if not os.path.exists(self.engine.model_path):
            raise FileNotFoundError("Trained model not found. Please train the model first.")
        
        torch.serialization.add_safe_globals(['sklearn.preprocessing._data.StandardScaler'])
        self.engine.model.load_state_dict(torch.load(self.engine.model_path, map_location=self.engine.device, weights_only=False))
        self.engine.model.eval()
        self.engine.scaler = joblib.load(self.engine.scaler_path)
        print("✅ Model and scaler loaded successfully")
    
    def prepare_test_data(self, test_size=0.2):
        print(f"{Fore.CYAN}Preparing test data...{Style.RESET_ALL}")
        
        all_sequences, all_targets, all_tickers = [], [], []
        
        for ticker in self.engine.tickers:
            ticker_data = self.data[self.data['ticker'] == ticker]
            ticker_features = ticker_data[self.engine.feature_cols]
            
            ticker_features_scaled = self.engine.scaler.transform(ticker_features)
            
            # Create sequences
            seq_len = config.MODEL_CONFIG['sequence_length']
            pred_hor = config.MODEL_CONFIG['prediction_horizon']
            
            for i in range(len(ticker_features_scaled) - seq_len - pred_hor + 1):
                sequence = ticker_features_scaled[i:i + seq_len]
                target = ticker_features_scaled[i + seq_len:i + seq_len + pred_hor]
                
                all_sequences.append(sequence)
                all_targets.append(target)
                all_tickers.append(ticker)
        
        # Convert to arrays
        X = np.array(all_sequences)
        y = np.array(all_targets)
        tickers = np.array(all_tickers)
        
        # Use the last test_size portion as test set (most recent data)
        test_start = int(len(X) * (1 - test_size))
        
        self.X_test = X[test_start:]
        self.y_test = y[test_start:]
        self.test_tickers = tickers[test_start:]
        
        print(f"✅ Test data prepared: {len(self.X_test)} samples")
    
    def predict_batch(self, X_batch, ticker_batch):
        predictions = []
        
        with torch.no_grad():
            batch_pbar = tqdm(range(len(X_batch)), desc="Making predictions", leave=False)
            for i in batch_pbar:
                seq = torch.FloatTensor(X_batch[i]).unsqueeze(0).to(self.engine.device)
                ticker_id = torch.LongTensor([self.engine.ticker_map[ticker_batch[i]]]).to(self.engine.device)
                
                pred = self.engine.model(seq, ticker_id)
                predictions.append(pred.cpu().numpy().squeeze())
        
        return np.array(predictions)
    
    def calculate_price_accuracy(self, y_true, y_pred):
        close_idx = self.engine.feature_cols.index(config.TARGET_FEATURE)
        
        true_returns = y_true[:, :, close_idx]
        pred_returns = y_pred[:, :, close_idx]
        
        # Convert log returns to actual returns
        true_returns_actual = np.expm1(true_returns)
        pred_returns_actual = np.expm1(pred_returns)
        
        # Calculate metrics
        mae = mean_absolute_error(true_returns_actual.flatten(), pred_returns_actual.flatten())
        mse = mean_squared_error(true_returns_actual.flatten(), pred_returns_actual.flatten())
        rmse = np.sqrt(mse)
        r2 = r2_score(true_returns_actual.flatten(), pred_returns_actual.flatten())
        
        # Direction accuracy (whether we predicted the right direction)
        true_direction = np.sign(true_returns_actual)
        pred_direction = np.sign(pred_returns_actual)
        direction_accuracy = np.mean(true_direction == pred_direction)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((true_returns_actual - pred_returns_actual) / (true_returns_actual + 1e-8))) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'Direction_Accuracy': direction_accuracy,
            'MAPE': mape
        }
    
    def calculate_trading_metrics(self, y_true, y_pred):
        close_idx = self.engine.feature_cols.index(config.TARGET_FEATURE)
        
        true_returns = y_true[:, 0, close_idx]
        pred_returns = y_pred[:, 0, close_idx]
        
        # Convert to actual returns
        true_returns_actual = np.expm1(true_returns)
        pred_returns_actual = np.expm1(pred_returns)
        
        # Simple trading strategy: buy if predicted return > threshold
        threshold = 0.001
        
        # Trading signals
        signals = (pred_returns_actual > threshold).astype(int)
        
        # Calculate returns if we follow the signals
        strategy_returns = signals * true_returns_actual
        
        # Performance metrics
        total_return = np.sum(strategy_returns)
        win_rate = np.mean(strategy_returns[signals == 1] > 0) if np.sum(signals) > 0 else 0
        sharpe_ratio = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
        
        return {
            'Total_Return': total_return,
            'Win_Rate': win_rate,
            'Sharpe_Ratio': sharpe_ratio,
            'Num_Trades': np.sum(signals)
        }
    
    def test_by_ticker(self):
        print(f"{Fore.CYAN}Testing accuracy by ticker...{Style.RESET_ALL}")
        
        ticker_results = {}
        
        ticker_pbar = tqdm(self.engine.tickers, desc="Testing tickers", leave=False)
        
        for ticker in ticker_pbar:
            ticker_mask = self.test_tickers == ticker
            if not np.any(ticker_mask):
                continue
                
            X_ticker = self.X_test[ticker_mask]
            y_ticker = self.y_test[ticker_mask]
            tickers_ticker = self.test_tickers[ticker_mask]
            
            if len(X_ticker) == 0:
                continue
            
            y_pred = self.predict_batch(X_ticker, tickers_ticker)
            
            price_metrics = self.calculate_price_accuracy(y_ticker, y_pred)
            trading_metrics = self.calculate_trading_metrics(y_ticker, y_pred)
            
            ticker_results[ticker] = {**price_metrics, **trading_metrics}
            ticker_pbar.set_postfix({ticker: f"R²={price_metrics['R²']:.3f}"})
        
        return ticker_results
    
    def run_comprehensive_test(self):
        print("🧪 Running comprehensive accuracy testing...")
        
        self.prepare_test_data()
        
        # Overall predictions
        print("🔮 Making predictions on test set...")
        y_pred = self.predict_batch(self.X_test, self.test_tickers)
        
        # Overall metrics
        print(f"{Fore.CYAN}Calculating overall metrics...{Style.RESET_ALL}")
        overall_price_metrics = self.calculate_price_accuracy(self.y_test, y_pred)
        overall_trading_metrics = self.calculate_trading_metrics(self.y_test, y_pred)
        
        # Per-ticker metrics
        print("Per-ticker metrics...")
        ticker_results = self.test_by_ticker()
        
        overall_metrics = {**overall_price_metrics, **overall_trading_metrics}
        
        results = {
            'overall_metrics': overall_metrics,
            'ticker_metrics': ticker_results,
            'test_samples': len(self.X_test),
            'accuracy_metrics': {
                'mae': overall_metrics.get('MAE', 0),
                'rmse': overall_metrics.get('RMSE', 0),
                'r2': overall_metrics.get('R²', 0),
                'mape': overall_metrics.get('MAPE', 0),
                'direction_accuracy': overall_metrics.get('Direction_Accuracy', 0)
            }
        }
        
        return results
    
    def print_results(self, results):
        print("\n" + "="*80)
        print("MODEL ACCURACY TEST RESULTS")
        print("="*80)
        
        print(f"\n{Fore.WHITE}OVERALL PERFORMANCE ({results['test_samples']} test samples){Style.RESET_ALL}")
        print("-" * 60)
        overall = results['overall_metrics']
        
        print(f"💰 Price Prediction Accuracy:")
        print(f"   • Mean Absolute Error (MAE):     {overall['MAE']:.4f}")
        print(f"   • Root Mean Square Error (RMSE): {overall['RMSE']:.4f}")
        print(f"   • R² Score:                      {overall['R²']:.4f}")
        print(f"   • Mean Absolute % Error (MAPE):  {overall['MAPE']:.2f}%")
        print(f"   • Direction Accuracy:            {overall['Direction_Accuracy']:.2%}")
        
        print(f"\n{Fore.GREEN}Trading Performance:{Style.RESET_ALL}")
        print(f"   • Total Return:                  {overall['Total_Return']:.4f}")
        print(f"   • Win Rate:                      {overall['Win_Rate']:.2%}")
        print(f"   • Sharpe Ratio:                  {overall['Sharpe_Ratio']:.4f}")
        print(f"   • Number of Trades:              {overall['Num_Trades']}")
        
        print(f"\n🎯 PER-TICKER PERFORMANCE")
        print("-" * 60)
        
        # Create summary table
        ticker_summary = []
        for ticker, metrics in results['ticker_metrics'].items():
            ticker_summary.append({
                'Ticker': ticker,
                'R²': f"{metrics['R²']:.3f}",
                'Direction_Acc': f"{metrics['Direction_Accuracy']:.1%}",
                'MAPE': f"{metrics['MAPE']:.1f}%",
                'Win_Rate': f"{metrics['Win_Rate']:.1%}",
                'Sharpe': f"{metrics['Sharpe_Ratio']:.2f}"
            })
        
        df_summary = pd.DataFrame(ticker_summary)
        print(df_summary.to_string(index=False))
        
        # Performance grades
        print(f"\n🏆 PERFORMANCE GRADES")
        print("-" * 30)
        
        r2_score = overall['R²']
        direction_acc = overall['Direction_Accuracy']
        mape = overall['MAPE']
        
        # Grade R²
        if r2_score >= 0.8:
            r2_grade = "A+ (Excellent)"
        elif r2_score >= 0.6:
            r2_grade = "A (Very Good)"
        elif r2_score >= 0.4:
            r2_grade = "B (Good)"
        elif r2_score >= 0.2:
            r2_grade = "C (Fair)"
        else:
            r2_grade = "D (Poor)"
        
        # Grade Direction Accuracy
        if direction_acc >= 0.65:
            dir_grade = "A+ (Excellent)"
        elif direction_acc >= 0.6:
            dir_grade = "A (Very Good)"
        elif direction_acc >= 0.55:
            dir_grade = "B (Good)"
        elif direction_acc >= 0.5:
            dir_grade = "C (Fair)"
        else:
            dir_grade = "D (Poor)"
        
        # Grade MAPE
        if mape <= 5:
            mape_grade = "A+ (Excellent)"
        elif mape <= 10:
            mape_grade = "A (Very Good)"
        elif mape <= 20:
            mape_grade = "B (Good)"
        elif mape <= 30:
            mape_grade = "C (Fair)"
        else:
            mape_grade = "D (Poor)"
        
        print(f"   • R² Score:           {r2_grade}")
        print(f"   • Direction Accuracy: {dir_grade}")
        print(f"   • MAPE:               {mape_grade}")
        
        print("\n" + "="*80)

    def get_best_worst_predictions(self, y_true, y_pred, test_tickers):
        close_idx = self.engine.feature_cols.index(config.TARGET_FEATURE)
        
        individual_scores = []
        for i in range(len(y_true)):
            true_seq = y_true[i, :, close_idx]
            pred_seq = y_pred[i, :, close_idx]
            
            # Calculate R² for this sequence
            r2 = r2_score(true_seq, pred_seq)
            individual_scores.append({
                'index': i,
                'r2': r2,
                'ticker': test_tickers[i],
                'true_returns': true_seq,
                'pred_returns': pred_seq
            })
        
        # Sort by R² score
        individual_scores.sort(key=lambda x: x['r2'], reverse=True)
        
        # Get best and worst
        best_predictions = individual_scores[:3]
        worst_predictions = individual_scores[-3:]
        
        return best_predictions, worst_predictions
    
    def plot_best_worst_predictions(self, results=None):
        if results is None:
            results = self.run_comprehensive_test()
        
        y_pred = self.predict_batch(self.X_test, self.test_tickers)
        
        best_preds, worst_preds = self.get_best_worst_predictions(self.y_test, y_pred, self.test_tickers)
        
        # Create the plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Predictions: Best vs Worst Performance', fontsize=16, fontweight='bold')
        
        # Plot best predictions
        for i, pred_info in enumerate(best_preds):
            ax = axes[0, i]
            
            # Convert log returns to price movements (cumulative)
            true_returns = pred_info['true_returns']
            pred_returns = pred_info['pred_returns']
            
            # Convert to actual returns
            true_actual = np.expm1(true_returns)
            pred_actual = np.expm1(pred_returns)
            
            # Calculate cumulative price movements (starting from 100)
            true_prices = 100 * np.cumprod(1 + true_actual)
            pred_prices = 100 * np.cumprod(1 + pred_actual)
            
            days = range(1, len(true_returns) + 1)
            
            ax.plot(days, true_prices, 'g-', linewidth=2, label='Actual', marker='o')
            ax.plot(days, pred_prices, 'b--', linewidth=2, label='Predicted', marker='s')
            
            ax.set_title(f'BEST #{i+1}: {pred_info["ticker"]}\nR² = {pred_info["r2"]:.4f}', 
                        fontweight='bold', color='green')
            ax.set_xlabel('Days')
            ax.set_ylabel('Price Movement (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add performance metrics as text
            mae = mean_absolute_error(true_actual, pred_actual)
            direction_acc = np.mean(np.sign(true_actual) == np.sign(pred_actual))
            ax.text(0.02, 0.98, f'MAE: {mae:.4f}\nDir Acc: {direction_acc:.2%}', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Plot worst predictions
        for i, pred_info in enumerate(worst_preds):
            ax = axes[1, i]
            
            # Convert log returns to price movements
            true_returns = pred_info['true_returns']
            pred_returns = pred_info['pred_returns']
            
            # Convert to actual returns
            true_actual = np.expm1(true_returns)
            pred_actual = np.expm1(pred_returns)
            
            # Calculate cumulative price movements (starting from 100)
            true_prices = 100 * np.cumprod(1 + true_actual)
            pred_prices = 100 * np.cumprod(1 + pred_actual)
            
            days = range(1, len(true_returns) + 1)
            
            ax.plot(days, true_prices, 'r-', linewidth=2, label='Actual', marker='o')
            ax.plot(days, pred_prices, 'orange', linestyle='--', linewidth=2, label='Predicted', marker='s')
            
            ax.set_title(f'WORST #{i+1}: {pred_info["ticker"]}\nR² = {pred_info["r2"]:.4f}', 
                        fontweight='bold', color='red')
            ax.set_xlabel('Days')
            ax.set_ylabel('Price Movement (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add performance metrics as text
            mae = mean_absolute_error(true_actual, pred_actual)
            direction_acc = np.mean(np.sign(true_actual) == np.sign(pred_actual))
            ax.text(0.02, 0.98, f'MAE: {mae:.4f}\nDir Acc: {direction_acc:.2%}', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Also create a summary comparison chart
        self.plot_prediction_distribution(results)
    
    def plot_prediction_distribution(self, results):
        """Plot the distribution of prediction accuracies"""
        # Get all individual R² scores
        y_pred = self.predict_batch(self.X_test, self.test_tickers)
        close_idx = self.engine.feature_cols.index(config.TARGET_FEATURE)
        
        individual_r2_scores = []
        for i in range(len(self.y_test)):
            true_seq = self.y_test[i, :, close_idx]
            pred_seq = y_pred[i, :, close_idx]
            r2 = r2_score(true_seq, pred_seq)
            individual_r2_scores.append(r2)
        
        # Create distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of R² scores
        axes[0].hist(individual_r2_scores, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].axvline(np.mean(individual_r2_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(individual_r2_scores):.4f}')
        axes[0].axvline(np.median(individual_r2_scores), color='green', linestyle='--', 
                       label=f'Median: {np.median(individual_r2_scores):.4f}')
        axes[0].set_xlabel('R² Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Individual Prediction R² Scores')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot by ticker
        ticker_r2_data = []
        ticker_labels = []
        for ticker in self.engine.tickers:
            ticker_mask = self.test_tickers == ticker
            if np.any(ticker_mask):
                ticker_r2 = [individual_r2_scores[i] for i in range(len(individual_r2_scores)) if ticker_mask[i]]
                ticker_r2_data.append(ticker_r2)
                ticker_labels.append(ticker)
        
        if ticker_r2_data:
            axes[1].boxplot(ticker_r2_data, labels=ticker_labels)
            axes[1].set_ylabel('R² Score')
            axes[1].set_title('R² Score Distribution by Ticker')
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        print(f"\n{Fore.CYAN}PREDICTION ACCURACY STATISTICS{Style.RESET_ALL}")
        print("-" * 50)
        print(f"Best R² Score:     {np.max(individual_r2_scores):.4f}")
        print(f"Worst R² Score:    {np.min(individual_r2_scores):.4f}")
        print(f"Mean R² Score:     {np.mean(individual_r2_scores):.4f}")
        print(f"Median R² Score:   {np.median(individual_r2_scores):.4f}")
        print(f"Std Dev R² Score:  {np.std(individual_r2_scores):.4f}")
        print(f"% Positive R²:     {(np.array(individual_r2_scores) > 0).mean():.2%}")
        
        return individual_r2_scores

    def save_prediction_plots(self, results=None, save_path="prediction_analysis.png"):
        """Save the prediction plots to file"""
        if results is None:
            results = self.run_comprehensive_test()
        
        # Set matplotlib backend to save plots
        plt.ioff()  # Turn off interactive mode
        
        # Get predictions
        y_pred = self.predict_batch(self.X_test, self.test_tickers)
        
        # Get best and worst predictions
        best_preds, worst_preds = self.get_best_worst_predictions(self.y_test, y_pred, self.test_tickers)
        
        # Create the plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Predictions: Best vs Worst Performance', fontsize=16, fontweight='bold')
        
        # Plot best predictions
        for i, pred_info in enumerate(best_preds):
            ax = axes[0, i]
            
            # Convert log returns to price movements (cumulative)
            true_returns = pred_info['true_returns']
            pred_returns = pred_info['pred_returns']
            
            # Convert to actual returns
            true_actual = np.expm1(true_returns)
            pred_actual = np.expm1(pred_returns)
            
            # Calculate cumulative price movements (starting from 100)
            true_prices = 100 * np.cumprod(1 + true_actual)
            pred_prices = 100 * np.cumprod(1 + pred_actual)
            
            days = range(1, len(true_returns) + 1)
            
            ax.plot(days, true_prices, 'g-', linewidth=2, label='Actual', marker='o')
            ax.plot(days, pred_prices, 'b--', linewidth=2, label='Predicted', marker='s')
            
            ax.set_title(f'BEST #{i+1}: {pred_info["ticker"]}\nR² = {pred_info["r2"]:.4f}', 
                        fontweight='bold', color='green')
            ax.set_xlabel('Days')
            ax.set_ylabel('Price Movement (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add performance metrics as text
            mae = mean_absolute_error(true_actual, pred_actual)
            direction_acc = np.mean(np.sign(true_actual) == np.sign(pred_actual))
            ax.text(0.02, 0.98, f'MAE: {mae:.4f}\nDir Acc: {direction_acc:.2%}', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Plot worst predictions
        for i, pred_info in enumerate(worst_preds):
            ax = axes[1, i]
            
            # Convert log returns to price movements
            true_returns = pred_info['true_returns']
            pred_returns = pred_info['pred_returns']
            
            # Convert to actual returns
            true_actual = np.expm1(true_returns)
            pred_actual = np.expm1(pred_returns)
            
            # Calculate cumulative price movements (starting from 100)
            true_prices = 100 * np.cumprod(1 + true_actual)
            pred_prices = 100 * np.cumprod(1 + pred_actual)
            
            days = range(1, len(true_returns) + 1)
            
            ax.plot(days, true_prices, 'r-', linewidth=2, label='Actual', marker='o')
            ax.plot(days, pred_prices, 'orange', linestyle='--', linewidth=2, label='Predicted', marker='s')
            
            ax.set_title(f'WORST #{i+1}: {pred_info["ticker"]}\nR² = {pred_info["r2"]:.4f}', 
                        fontweight='bold', color='red')
            ax.set_xlabel('Days')
            ax.set_ylabel('Price Movement (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add performance metrics as text
            mae = mean_absolute_error(true_actual, pred_actual)
            direction_acc = np.mean(np.sign(true_actual) == np.sign(pred_actual))
            ax.text(0.02, 0.98, f'MAE: {mae:.4f}\nDir Acc: {direction_acc:.2%}', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{Fore.GREEN}Prediction analysis plot saved to: {save_path}{Style.RESET_ALL}")
        
        # Also save distribution plot
        dist_path = save_path.replace('.png', '_distribution.png')
        self.save_distribution_plot(results, dist_path)
        
        return save_path, dist_path
    
    def save_distribution_plot(self, results, save_path):
        """Save the distribution plot"""
        # Get all individual R² scores
        y_pred = self.predict_batch(self.X_test, self.test_tickers)
        close_idx = self.engine.feature_cols.index(config.TARGET_FEATURE)
        
        individual_r2_scores = []
        for i in range(len(self.y_test)):
            true_seq = self.y_test[i, :, close_idx]
            pred_seq = y_pred[i, :, close_idx]
            r2 = r2_score(true_seq, pred_seq)
            individual_r2_scores.append(r2)
        
        # Create distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of R² scores
        axes[0].hist(individual_r2_scores, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].axvline(np.mean(individual_r2_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(individual_r2_scores):.4f}')
        axes[0].axvline(np.median(individual_r2_scores), color='green', linestyle='--', 
                       label=f'Median: {np.median(individual_r2_scores):.4f}')
        axes[0].set_xlabel('R² Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Individual Prediction R² Scores')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot by ticker
        ticker_r2_data = []
        ticker_labels = []
        for ticker in self.engine.tickers:
            ticker_mask = self.test_tickers == ticker
            if np.any(ticker_mask):
                ticker_r2 = [individual_r2_scores[i] for i in range(len(individual_r2_scores)) if ticker_mask[i]]
                ticker_r2_data.append(ticker_r2)
                ticker_labels.append(ticker)
        
        if ticker_r2_data:
            axes[1].boxplot(ticker_r2_data, labels=ticker_labels)
            axes[1].set_ylabel('R² Score')
            axes[1].set_title('R² Score Distribution by Ticker')
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{Fore.GREEN}Distribution analysis plot saved to: {save_path}{Style.RESET_ALL}")

def main():
    global last_results
    try:
        print(f"{Fore.CYAN}Starting Model Accuracy Testing...{Style.RESET_ALL}")
        
        tester = AccuracyTester()
        results = tester.run_comprehensive_test()
        last_results = results
        tester.print_results(results)
        
        print(f"\n{Fore.CYAN}Generating visualization of best and worst predictions...{Style.RESET_ALL}")
        tester.plot_best_worst_predictions(results)
        
        plot_files = [
            os.path.join(config.MODELS_PATH, "prediction_analysis.png"),
            os.path.join(config.MODELS_PATH, "r2_distribution.png")
        ]
        
        print(f"\n{Fore.GREEN}Accuracy testing completed successfully!{Style.RESET_ALL}")
        
        return plot_files, results
        
    except Exception as e:
        print(f"{Fore.RED}Error during testing: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return [], None

