import sys
import os
import logging
import joblib 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config, utils
from core.data_loader import DataLoader
from core.trading_core import TradingCore
from core.rl_env import CryptoEnv
from models.hybrid_models import HybridLearner

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# [가속화] TensorFlow Mixed Precision 적용
from tensorflow.keras import mixed_precision # type: ignore
mixed_precision.set_global_policy('mixed_float16')

utils.silence_noisy_loggers()

class RLLoggingCallback(BaseCallback):
    def __init__(self, logger, verbose=0):
        super(RLLoggingCallback, self).__init__(verbose)
        self.custom_logger = logger

    def _on_training_start(self) -> None:
        self.custom_logger.info(f"   [RL] Training START (Total: {self.model._total_timesteps} steps)")

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            rew = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
            len_ = np.mean([ep['l'] for ep in self.model.ep_info_buffer])
            self.custom_logger.info(f"   [RL] Training END | Mean Ep_Reward: {rew:.2f} | Mean Ep_Len: {len_:.0f}")

class PipelineTrainer:
    def __init__(self, session_paths):
        self.paths = session_paths
        self.logger = utils.get_logger("Trainer", log_file=os.path.join(session_paths['root'], 'trainer.log'))
        self.loader = DataLoader(self.logger)
        self.model_dir = self.paths['model']
        # Scaler는 Loop 내부에서 매번 새로 생성하거나 fit 해야 함
        self.scaler = None 

    def log(self, msg):
        self.logger.info(msg)
        
    def _get_optimal_n_envs(self):
        return config.SYSTEM['NUM_WORKERS']

    def run_all(self):
        self.log(f"🚀 PIPELINE START: Session {self.paths['id']}")
        self.log("⚡ Acceleration Enabled: Mixed Precision(FP16) + Batch(1024)")
        
        # 1. Load Data
        self.log(f"[Phase 1] Loading Data...")
        full_df = self.loader.get_ml_data(config.MAIN_SYMBOL)
        if full_df.empty:
            self.log("!! [Error] No data found.")
            return

        train_idx_mask = full_df.index < config.TEST_SPLIT_DATE
        train_df = full_df[train_idx_mask].copy()
        test_df = full_df[~train_idx_mask].copy()
        
        feature_cols = [c for c in full_df.columns if c not in config.EXCLUDE_COLS]
        
        # 2. Signals (Walk-Forward Validation) - [Strict Mode]
        self.log("[Phase 2] Generating ML Signals (Strict Walk-Forward)...")
        full_df['ml_signal'] = 0.0 
        
        tscv = TimeSeriesSplit(n_splits=5)
        fold = 1
        
        for tr_idx, val_idx in tscv.split(train_df):
            self.log(f"   >> Fold {fold}/5...")
            fold_train = train_df.iloc[tr_idx]
            fold_val = train_df.iloc[val_idx]
            
            # [수정] Scaler Leakage 방지: Fold마다 Scaler를 새로 학습
            fold_scaler = RobustScaler()
            fold_scaler.fit(fold_train[feature_cols])
            
            # Fold별 데이터 변환
            X_flat_tr, X_seq_tr, y_tr = self._prepare_ml_inputs(fold_train, feature_cols, fold_scaler, is_training=True)
            X_flat_val, X_seq_val, _ = self._prepare_ml_inputs(fold_val, feature_cols, fold_scaler, is_training=False)
            
            if len(X_flat_tr) == 0 or len(X_flat_val) == 0: continue

            temp_model = HybridLearner(self.model_dir, self.logger)
            temp_model.train(X_flat_tr, y_tr, X_seq_tr, y_tr)
            
            signals = temp_model.predict_proba(X_flat_val, X_seq_val)
            
            valid_len = len(signals)
            target_idx = fold_val.index[config.ML_SEQ_LEN : config.ML_SEQ_LEN + valid_len]
            if not target_idx.empty:
                 full_df.loc[target_idx, 'ml_signal'] = signals[:len(target_idx)]
            
            fold += 1
            del temp_model
            del fold_scaler # 메모리 해제
            import gc; gc.collect()

        # 3. Final Training & Saving Scaler
        self.log("[Phase 3] Final Training (All Train Data)...")
        
        # 전체 Train 데이터로 Scaler 재학습 및 저장 (이게 최종 Scaler가 됨)
        self.scaler = RobustScaler()
        self.scaler.fit(train_df[feature_cols])
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        self.log("   - Scaler Saved.")

        # 최종 모델 학습
        X_flat_all, X_seq_all, y_all = self._prepare_ml_inputs(train_df, feature_cols, self.scaler, is_training=True)
        final_model = HybridLearner(self.model_dir, self.logger)
        final_model.train(X_flat_all, y_all, X_seq_all, y_all)
        
        # Test 데이터 예측
        X_flat_test, X_seq_test, _ = self._prepare_ml_inputs(test_df, feature_cols, self.scaler, is_training=False)
        test_signals = final_model.predict_proba(X_flat_test, X_seq_test)
        
        test_valid_len = len(test_signals)
        test_target_idx = test_df.index[config.ML_SEQ_LEN : config.ML_SEQ_LEN + test_valid_len]
        if not test_target_idx.empty:
            full_df.loc[test_target_idx, 'ml_signal'] = test_signals[:len(test_target_idx)]

        full_df['ml_signal'] = full_df['ml_signal'].fillna(0.0)

        # 4. RL Training
        self.log("[Phase 4] Training RL Agent...")
        self._train_rl(full_df[train_idx_mask])

        # 5. Backtest
        self.log("[Phase 5] Running Backtest...")
        precision_df = self.loader.get_precision_data(config.MAIN_SYMBOL)
        if not precision_df.empty:
            precision_df = precision_df[precision_df.index >= config.TEST_SPLIT_DATE]
        self._run_backtest(full_df[~train_idx_mask], precision_df)
        
        self.log("✅ PIPELINE FINISHED.")

    # [수정] scaler를 인자로 받도록 변경
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
        def make_env(): 
            utils.silence_noisy_loggers()
            return CryptoEnv(df, TradingCore(), precision_df=None, debug=False)
            
        n_envs = self._get_optimal_n_envs()
        env = SubprocVecEnv([make_env for _ in range(n_envs)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=config.RL_PPO_PARAMS['gamma'])
        
        checkpoint_callback = CheckpointCallback(save_freq=max(100_000 // n_envs, 1), save_path=self.paths['model'], name_prefix='rl_model')
        logging_callback = RLLoggingCallback(self.logger)

        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=self.paths['tb'], device=config.SYSTEM['MAIN_RL_DEVICE'], **config.RL_PPO_PARAMS)
        model.learn(total_timesteps=config.RL_TOTAL_TIMESTEPS, callback=[checkpoint_callback, logging_callback], progress_bar=True)
        model.save(os.path.join(self.paths['model'], "final_agent"))
        env.save(os.path.join(self.paths['model'], "vec_normalize.pkl"))
        env.close()

    def _run_backtest(self, df, precision_df):
        env = CryptoEnv(df, TradingCore(), precision_df=precision_df, debug=True)
        dummy_env = DummyVecEnv([lambda: env])
        vec_norm_path = os.path.join(self.paths['model'], "vec_normalize.pkl")
        
        if os.path.exists(vec_norm_path):
            norm_env = VecNormalize.load(vec_norm_path, dummy_env)
            norm_env.training = False 
            norm_env.norm_reward = False
        else: norm_env = dummy_env

        model = PPO.load(os.path.join(self.paths['model'], "final_agent"))
        obs = norm_env.reset()
        done = [False]
        history_dates, history_bal, history_price = [], [], []
        final_bal = config.INITIAL_BALANCE
        
        for i in tqdm(range(len(df)), desc="Backtesting"):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = norm_env.step(action)
            if done[0]:
                final_bal = infos[0]['final_balance']
                break
            try:
                real_env = norm_env.envs[0]
                history_bal.append(real_env.logic.balance)
                history_price.append(real_env.df.iloc[real_env.current_step]['close'])
                history_dates.append(df.index[i])
            except: pass

        save_path = os.path.join(self.paths['root'], 'backtest_result.png')
        self._plot_backtest(history_dates, history_bal, history_price, save_path)
        real_env = norm_env.envs[0]
        if real_env.logic.history:
            pd.DataFrame(real_env.logic.history).to_csv(os.path.join(self.paths['root'], 'trade_history.csv'), index=False)
        
        roi = (final_bal - config.INITIAL_BALANCE) / config.INITIAL_BALANCE * 100
        self.log(f"   - Final Balance: ${final_bal:,.2f} (ROI: {roi:+.2f}%)")

    def _plot_backtest(self, dates, balances, prices, save_path):
        if not dates or not balances: return
        try:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(dates, balances, color='blue', label='Balance')
            ax2 = ax1.twinx()
            ax2.plot(dates, prices, color='gray', alpha=0.3, label='Price')
            plt.title('Backtest Result')
            plt.savefig(save_path)
            plt.close()
        except: pass