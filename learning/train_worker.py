import sys
import os
import logging
import joblib 
import numpy as np
import pandas as pd
import subprocess
from tqdm import tqdm
import traceback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import constants as config
from core import utils
from core.data_processor import DataProcessor
from core.config_manager import ConfigManager
from learning.networks import HybridLearner
from execution.trade_engine import TradingCore
from learning.rl_environment import CryptoEnv

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from tensorflow.keras import mixed_precision # type: ignore
mixed_precision.set_global_policy('mixed_float16')

utils.silence_noisy_loggers()

class TrainWorker:
    def __init__(self, session_paths, config_override=None):
        self.paths = session_paths
        self.logger = utils.get_logger("TrainWorker", log_file=os.path.join(session_paths['root'], 'trainer.log'))
        self.processor = DataProcessor(self.logger)
        self.model_dir = self.paths['model']
        self.scaler = None
        
        if config_override:
            self.conf = config_override
            self.logger.info("⚙️ Using Overridden Configuration.")
        else:
            cl = ConfigManager(config.MAIN_SYMBOL)
            self.conf = cl.load_config()
            self.logger.info("⚙️ Using Default Configuration.")

    def log(self, msg):
        self.logger.info(msg)
        
    def _get_optimal_n_envs(self):
        return config.SYSTEM['NUM_WORKERS']

    def run_all(self):
        self.log(f"🚀 TRAINING START: Session {self.paths['id']}")
        
        self.log(f"[Phase 1] Loading Data...")
        full_df = self.processor.get_ml_data(config.MAIN_SYMBOL)
        if full_df.empty:
            self.log("!! [Error] No data found.")
            return

        train_idx_mask = full_df.index < config.TEST_SPLIT_DATE
        train_df = full_df[train_idx_mask].copy()
        test_df = full_df[~train_idx_mask].copy()
        
        feature_cols = [c for c in full_df.columns if c not in config.EXCLUDE_COLS]
        
        self.log("[Phase 2] Generating ML Signals...")
        full_df['ml_signal'] = 0.0 
        
        tscv = TimeSeriesSplit(n_splits=5)
        fold = 1
        
        if 'strategy_trend' in self.conf:
            xgb_params = self.conf['strategy_trend'].get('xgb_params', config.XGB_PARAMS)
            lstm_params = self.conf['strategy_trend'].get('lstm_params', config.LSTM_PARAMS)
        else:
            xgb_params = self.conf.get('xgb_params', config.XGB_PARAMS)
            lstm_params = self.conf.get('lstm_params', config.LSTM_PARAMS)

        for tr_idx, val_idx in tscv.split(train_df):
            fold_train = train_df.iloc[tr_idx]
            fold_val = train_df.iloc[val_idx]
            
            fold_scaler = RobustScaler()
            fold_scaler.fit(fold_train[feature_cols])
            
            X_flat_tr, X_seq_tr, y_tr = self._prepare_ml_inputs(fold_train, feature_cols, fold_scaler, is_training=True)
            X_flat_val, X_seq_val, _ = self._prepare_ml_inputs(fold_val, feature_cols, fold_scaler, is_training=False)
            
            if len(X_flat_tr) == 0 or len(X_flat_val) == 0: continue

            temp_model = HybridLearner(self.model_dir, self.logger)
            temp_model.xgb_params = xgb_params
            temp_model.lstm_params = lstm_params
            
            temp_model.train(X_flat_tr, y_tr, X_seq_tr, y_tr)
            signals = temp_model.predict_proba(X_flat_val, X_seq_val)
            
            valid_len = len(signals)
            target_idx = fold_val.index[config.ML_SEQ_LEN : config.ML_SEQ_LEN + valid_len]
            
            if not target_idx.empty:
                if signals.ndim > 1:
                    insert_values = signals[:len(target_idx), 1] - signals[:len(target_idx), 0]
                else:
                    insert_values = signals[:len(target_idx)]
                full_df.loc[target_idx, 'ml_signal'] = insert_values
            
            fold += 1
            del temp_model
            del fold_scaler
            import gc; gc.collect()

        self.log("[Phase 3] Final Training...")
        self.scaler = RobustScaler()
        self.scaler.fit(train_df[feature_cols])
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.pkl"))

        X_flat_all, X_seq_all, y_all = self._prepare_ml_inputs(train_df, feature_cols, self.scaler, is_training=True)
        
        final_model = HybridLearner(self.model_dir, self.logger)
        final_model.xgb_params = xgb_params
        final_model.lstm_params = lstm_params
        final_model.train(X_flat_all, y_all, X_seq_all, y_all)
        
        X_flat_test, X_seq_test, _ = self._prepare_ml_inputs(test_df, feature_cols, self.scaler, is_training=False)
        test_signals = final_model.predict_proba(X_flat_test, X_seq_test)
        
        test_valid_len = len(test_signals)
        test_target_idx = test_df.index[config.ML_SEQ_LEN : config.ML_SEQ_LEN + test_valid_len]
        
        if not test_target_idx.empty:
            if test_signals.ndim > 1:
                insert_values = test_signals[:len(test_target_idx), 1] - test_signals[:len(test_target_idx), 0]
            else:
                insert_values = test_signals[:len(test_target_idx)]
            full_df.loc[test_target_idx, 'ml_signal'] = insert_values

        full_df['ml_signal'] = full_df['ml_signal'].fillna(0.0)

        # 4. RL Training (Skipped for Manager Phase)
        self.log("[Phase 4] Skipping RL Agent Training (Waiting for Manager Phase)...")
        # self._train_rl(full_df[train_idx_mask])

        # 5. Backtest
        self.log("[Phase 5] Running Backtest...")
        precision_df = self.processor.get_precision_data(config.MAIN_SYMBOL)
        if not precision_df.empty:
            precision_df = precision_df[precision_df.index >= config.TEST_SPLIT_DATE]
        self._run_backtest(full_df[~train_idx_mask], precision_df)
        
        self.log("✅ TRAINING PIPELINE FINISHED.")

    def _prepare_ml_inputs(self, df, features, scaler, is_training=False):
        data_scaled = scaler.transform(df[features])
        if len(data_scaled) <= config.ML_SEQ_LEN: return [], [], []
        
        X_seq = np.lib.stride_tricks.sliding_window_view(data_scaled, window_shape=(config.ML_SEQ_LEN, len(features)))
        if X_seq.ndim == 4: X_seq = X_seq.squeeze(axis=1)
        X_flat = data_scaled[config.ML_SEQ_LEN:]
        
        min_len = min(len(X_seq), len(X_flat))
        
        if 'target_cls' in df.columns and is_training:
            y = df['target_cls'].values[config.ML_SEQ_LEN:]
            min_len = min(min_len, len(y))
            return X_flat[:min_len], X_seq[:min_len], y[:min_len]
        else:
            return X_flat[:min_len], X_seq[:min_len], None

    def _train_rl(self, df):
        # ... (기존 코드 유지) ...
        pass

    def _run_backtest(self, df, precision_df):
        core = TradingCore('strategy_trend')
        env = CryptoEnv(df, core, precision_df=precision_df, debug=True)
        dummy_env = DummyVecEnv([lambda: env])
        
        model_path = os.path.join(self.paths['model'], "final_agent")
        # RL 모델이 없으면 백테스트 불가하므로 패스 (현재는 스킵)
        if not os.path.exists(model_path + ".zip"):
            self.log("⚠️ RL Agent model not found. Skipping RL backtest.")
            return

        model = PPO.load(model_path)
        obs = dummy_env.reset()
        done = [False]
        final_bal = config.INITIAL_BALANCE
        
        for i in tqdm(range(len(df)), desc="Backtesting"):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = dummy_env.step(action)
            if done[0]:
                final_bal = infos[0]['final_balance']
                break
        
        roi = (final_bal - config.INITIAL_BALANCE) / config.INITIAL_BALANCE * 100
        self.log(f"   - Final Balance: ${final_bal:,.2f} (ROI: {roi:+.2f}%)")

    def _plot_backtest(self, dates, balances, prices, save_path):
        pass # 생략

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    temp_logger = utils.get_logger("TrainMain")
    
    from core.constants import SessionManager
    sm = SessionManager()
    
    # Factory 실행용 환경변수 체크
    factory_sid = os.environ.get('FACTORY_SESSION_ID')
    
    try:
        if factory_sid:
            sm.session_id = factory_sid
            sm.log_dir = os.path.join(config.LOG_BASE_DIR, factory_sid)
            sm.model_dir = os.path.join(config.MODEL_BASE_DIR, factory_sid)
            sm.tensorboard_dir = os.path.join(sm.log_dir, "tb_logs")
            paths = {
                'id': sm.session_id, 'root': sm.log_dir,
                'tb': sm.tensorboard_dir, 'model': sm.model_dir,
                'log_file': os.path.join(sm.log_dir, 'trainer.log')
            }
            temp_logger.info(f"🔗 Linked to Factory Session: {factory_sid}")
            trainer = TrainWorker(paths)
            trainer.run_all()
        else:
            paths = sm.create()
            temp_logger.info(f"🆕 Created New Session: {paths['id']}")
            trainer = TrainWorker(paths)
            trainer.run_all()
            
    except Exception as e:
        temp_logger.error(f"\n❌ Trainer Crashed: {e}")
        temp_logger.error(traceback.format_exc())