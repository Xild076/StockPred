import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    from .modeling.engine import UniversalModelEngine
    from .modeling.architecture import UniversalTransformerModel
    from . import config
    from .model_manager import ModelManager
except (ImportError, ValueError):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.modeling.engine import UniversalModelEngine
    from src.modeling.architecture import UniversalTransformerModel
    from src import config
    from src.model_manager import ModelManager

class AccuracyTester:
    def __init__(self, model_path=None):
        self.model_manager = ModelManager()
        if model_path:
            self.model_dir = model_path
        else:
            saved_models = self.model_manager.list_saved_models()
            if not saved_models:
                raise FileNotFoundError("No trained models found.")
            self.model_dir = saved_models[0]['path']
        
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.model_config = self.metadata.get('model_config', config.MODEL_CONFIG)
        self.feature_cols = self.metadata.get('feature_columns', [])
        self.tickers = self.metadata.get('tickers', [])

        self.data = pd.read_parquet(config.DATA_PATH)
        self.engine = UniversalModelEngine(self.data, model_config=self.model_config)
        self.engine.feature_cols = self.feature_cols
        self.engine.tickers = self.tickers
        self.engine.ticker_map = {ticker: i for i, ticker in enumerate(self.tickers)}

        self.load_trained_model()

    def load_trained_model(self):
        model_path = os.path.join(self.model_dir, 'model.pth')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model or scaler not found in {self.model_dir}")

        supported_params = {
            'd_model', 'n_heads', 'n_layers', 'dropout', 'ticker_embedding_dim',
            'sequence_length', 'prediction_horizon'
        }
        filtered_config = {k: v for k, v in self.model_config.items() if k in supported_params}

        self.engine.model = UniversalTransformerModel(
            input_dim=len(self.feature_cols),
            ticker_count=len(self.tickers),
            **filtered_config
        ).to(self.engine.device)

        state_dict = torch.load(model_path, map_location=self.engine.device, weights_only=False)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        self.engine.model.load_state_dict(state_dict)
        self.engine.model.eval()
        self.engine.scaler = joblib.load(scaler_path)

    def prepare_test_data(self, test_size=0.2):
        all_sequences, all_targets, all_tickers = [], [], []

        for ticker in self.tickers:
            ticker_data = self.data[self.data['ticker'] == ticker]
            if ticker_data.empty:
                continue

            ticker_features = ticker_data[self.feature_cols]
            ticker_features_scaled = self.engine.scaler.transform(ticker_features)

            seq_len = self.model_config['sequence_length']
            pred_hor = self.model_config['prediction_horizon']

            for i in range(len(ticker_features_scaled) - seq_len - pred_hor + 1):
                sequence = ticker_features_scaled[i:i + seq_len]
                target = ticker_features_scaled[i + seq_len:i + seq_len + pred_hor]
                all_sequences.append(sequence)
                all_targets.append(target)
                all_tickers.append(ticker)

        X = np.array(all_sequences)
        y = np.array(all_targets)
        tickers = np.array(all_tickers)

        test_start = int(len(X) * (1 - test_size))
        self.X_test = X[test_start:]
        self.y_test = y[test_start:]
        self.test_tickers = tickers[test_start:]

    def predict_batch(self, X_batch, ticker_batch):
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(len(X_batch)), desc="Making predictions", leave=False):
                seq = torch.FloatTensor(X_batch[i]).unsqueeze(0).to(self.engine.device)
                ticker_id = torch.LongTensor([self.engine.ticker_map[ticker_batch[i]]]).to(self.engine.device)
                pred = self.engine.model(seq, ticker_id)
                predictions.append(pred.cpu().numpy().squeeze())
        return np.array(predictions)

    def calculate_price_accuracy(self, y_true, y_pred):
        close_idx = self.feature_cols.index(config.TARGET_FEATURE)
        true_returns = y_true[:, :, close_idx]
        pred_returns = y_pred[:, :, close_idx]

        true_returns_actual = np.expm1(true_returns)
        pred_returns_actual = np.expm1(pred_returns)

        mae = mean_absolute_error(true_returns_actual.flatten(), pred_returns_actual.flatten())
        mse = mean_squared_error(true_returns_actual.flatten(), pred_returns_actual.flatten())
        rmse = np.sqrt(mse)
        r2 = r2_score(true_returns_actual.flatten(), pred_returns_actual.flatten())

        true_direction = np.sign(true_returns_actual)
        pred_direction = np.sign(pred_returns_actual)
        direction_accuracy = np.mean(true_direction == pred_direction)

        mape = np.mean(np.abs((true_returns_actual - pred_returns_actual) / (true_returns_actual + 1e-8))) * 100

        return {
            'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R²': r2,
            'Direction_Accuracy': direction_accuracy, 'MAPE': mape
        }

    def run_comprehensive_test(self):
        self.prepare_test_data()
        y_pred = self.predict_batch(self.X_test, self.test_tickers)
        
        overall_metrics = self.calculate_price_accuracy(self.y_test, y_pred)
        
        ticker_metrics = {}
        for ticker in self.tickers:
            mask = self.test_tickers == ticker
            if np.any(mask):
                ticker_metrics[ticker] = self.calculate_price_accuracy(self.y_test[mask], y_pred[mask])

        return {
            'overall_metrics': overall_metrics,
            'ticker_metrics': ticker_metrics,
            'test_samples': len(self.X_test),
            'y_true': self.y_test,
            'y_pred': y_pred,
            'test_tickers': self.test_tickers
        }

    def get_best_worst_predictions(self, results):
        close_idx = self.feature_cols.index(config.TARGET_FEATURE)
        y_true = results['y_true']
        y_pred = results['y_pred']
        test_tickers = results['test_tickers']
        
        scores = []
        for i in range(len(y_true)):
            r2 = r2_score(y_true[i, :, close_idx], y_pred[i, :, close_idx])
            scores.append({'index': i, 'r2': r2, 'ticker': test_tickers[i]})
        
        scores.sort(key=lambda x: x['r2'], reverse=True)
        
        best = scores[:3]
        worst = scores[-3:]
        
        return best, worst

    def plot_predictions(self, results, save_dir=None):
        best, worst = self.get_best_worst_predictions(results)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=150)
        fig.suptitle('Best and Worst Predictions', fontsize=16)

        plot_data = {'Best': best, 'Worst': worst}
        for i, (title, data) in enumerate(plot_data.items()):
            for j, score_info in enumerate(data):
                ax = axes[i, j]
                idx = score_info['index']
                ticker = score_info['ticker']
                r2 = score_info['r2']

                close_idx = self.feature_cols.index(config.TARGET_FEATURE)
                true_returns = results['y_true'][idx, :, close_idx]
                pred_returns = results['y_pred'][idx, :, close_idx]

                true_prices = 100 * np.cumprod(1 + np.expm1(true_returns))
                pred_prices = 100 * np.cumprod(1 + np.expm1(pred_returns))

                ax.plot(true_prices, label='Actual', marker='o')
                ax.plot(pred_prices, label='Predicted', marker='x', linestyle='--')
                ax.set_title(f'{title} #{j+1}: {ticker} (R²: {r2:.3f})')
                ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'best_worst_predictions.png'))
            plt.close()
        else:
            plt.show()

    def plot_best_worst_predictions(self, results):
        self.plot_predictions(results)
        self.plot_prediction_distribution(results)
        
    def save_prediction_plots(self, results, output_path):
        base_name = os.path.splitext(output_path)[0]
        save_dir = os.path.dirname(output_path) or '.'
        
        self.plot_predictions(results, save_dir)
        self.plot_prediction_distribution(results, save_dir)
        
        main_plot = os.path.join(save_dir, 'best_worst_predictions.png')
        dist_plot = os.path.join(save_dir, 'r2_distribution.png')
        
        return main_plot, dist_plot

    def plot_prediction_distribution(self, results, save_dir=None):
        close_idx = self.feature_cols.index(config.TARGET_FEATURE)
        y_true = results['y_true']
        y_pred = results['y_pred']
        r2_scores = [r2_score(y_true[i, :, close_idx], y_pred[i, :, close_idx]) for i in range(len(y_true))]

        plt.figure(figsize=(10, 6))
        sns.histplot(r2_scores, bins=50, kde=True)
        plt.title('Distribution of R² Scores')
        plt.xlabel('R² Score')
        plt.ylabel('Frequency')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'r2_distribution.png'))
            plt.close()
        else:
            plt.show()

def main():
    try:
        print("Starting comprehensive model accuracy testing...")
        
        tester = AccuracyTester()
        results = tester.run_comprehensive_test()
        
        overall_metrics = results['overall_metrics']
        ticker_metrics = results['ticker_metrics']
        
        print(f"\nOverall Model Performance:")
        print(f"MAE: {overall_metrics['MAE']:.4f}")
        print(f"RMSE: {overall_metrics['RMSE']:.4f}")
        print(f"R²: {overall_metrics['R²']:.4f}")
        print(f"Direction Accuracy: {overall_metrics['Direction_Accuracy']:.2%}")
        print(f"MAPE: {overall_metrics['MAPE']:.2f}%")
        
        print(f"\nTicker-Specific Performance:")
        for ticker, metrics in ticker_metrics.items():
            print(f"{ticker}: R²={metrics['R²']:.3f}, Dir={metrics['Direction_Accuracy']:.2%}")
        
        output_dir = os.path.join(tester.model_dir, 'analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        main_plot, dist_plot = tester.save_prediction_plots(results, os.path.join(output_dir, 'analysis.png'))
        
        accuracy_metrics = {
            'mae': float(overall_metrics['MAE']),
            'rmse': float(overall_metrics['RMSE']),
            'r2': float(overall_metrics['R²']),
            'direction_accuracy': float(overall_metrics['Direction_Accuracy']),
            'mape': float(overall_metrics['MAPE'])
        }
        
        metrics_file = os.path.join(output_dir, 'accuracy_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(accuracy_metrics, f, indent=2)
        
        print(f"\nAnalysis saved to: {output_dir}")
        print(f"Plots: {main_plot}, {dist_plot}")
        print(f"Metrics: {metrics_file}")
        
        return [main_plot, dist_plot, metrics_file], {'accuracy_metrics': accuracy_metrics}
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()

