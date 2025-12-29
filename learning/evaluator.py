import sys
import os
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import constants as config
from core import utils
from core.data_processor import DataProcessor
from execution.trade_engine import TradingCore
from learning.networks import HybridLearner
from sklearn.preprocessing import RobustScaler

utils.silence_noisy_loggers()

class ModelEvaluator:
    def __init__(self):
        self.logger = utils.get_logger("Evaluator")
        self.processor = DataProcessor(self.logger)

    def load_validation_data(self):
        full_df = self.processor.get_ml_data(config.MAIN_SYMBOL)
        val_df = full_df[full_df.index >= config.TEST_SPLIT_DATE].copy()
        return val_df

    def evaluate_model(self, model_dir, config_override=None):
        try:
            df = self.load_validation_data()
            feat_cols = [c for c in df.columns if c not in config.EXCLUDE_COLS]
            
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                data_scaled = scaler.transform(df[feat_cols])
            else:
                self.logger.warning(f"⚠️ Scaler not found in {model_dir}. Fitting new one.")
                scaler = RobustScaler()
                data_scaled = scaler.fit_transform(df[feat_cols])
            
            ml_model = HybridLearner(model_dir)
            
            X_seq = np.lib.stride_tricks.sliding_window_view(data_scaled, window_shape=(config.ML_SEQ_LEN, len(feat_cols)))
            if X_seq.ndim == 4: X_seq = X_seq.squeeze(axis=1)
            X_flat = data_scaled[config.ML_SEQ_LEN:]
            
            min_len = min(len(X_seq), len(X_flat))
            
            if not ml_model.load():
                self.logger.error("❌ Failed to load ML models.")
                return -999, 0
                
            signals = ml_model.predict_proba(X_flat[:min_len], X_seq[:min_len])
            if signals.ndim > 1: final_sig = signals[:, 1] - signals[:, 0]
            else: final_sig = signals

            sim_df = df.iloc[config.ML_SEQ_LEN:].iloc[:min_len].copy()
            sim_df['ml_signal'] = final_sig

            core = TradingCore('strategy_trend')
            
            if config_override and 'strategy_trend' in config_override:
                new_params = config_override['strategy_trend'].get('trading_rules', {})
                if new_params:
                    core.strategy.params.update(new_params)
            
            for i in range(len(sim_df)):
                row = sim_df.iloc[i]
                threshold = core.strategy.params.get('threshold_entry', config.TARGET_THRESHOLD)
                
                action = 0
                sig = row['ml_signal']
                if sig > threshold: action = 1
                elif sig < -threshold: action = 2
                
                core.process_step(action, row, row.name)
            
            final_balance = core.balance
            roi = (final_balance - config.INITIAL_BALANCE) / config.INITIAL_BALANCE * 100
            
            return roi, final_balance

        except Exception as e:
            self.logger.error(f"Evaluation Failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return -999, 0