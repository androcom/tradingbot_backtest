import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config

import logging
import joblib # [추가] Scaler 로드용
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sklearn.preprocessing import RobustScaler

from core.data_loader import DataLoader
from core.trading_core import TradingCore
from core.rl_env import CryptoEnv
from models.hybrid_models import HybridLearner

class ModelEvaluator:
    def __init__(self):
        self.logger = logging.getLogger("Evaluator")
        self.loader = DataLoader(self.logger)

    def load_validation_data(self):
        """최근 데이터 로드 (검증용, Test Split 이후)"""
        full_df = self.loader.get_ml_data(config.MAIN_SYMBOL)
        # 테스트 스플릿 이후의 데이터만 사용
        val_df = full_df[full_df.index >= config.TEST_SPLIT_DATE].copy()
        return val_df

    def evaluate_model(self, model_dir, df):
        """특정 모델의 수익률 계산"""
        try:
            # -----------------------------------------------------
            # [수정] Scaler 로드 (Data Leakage 방지)
            # -----------------------------------------------------
            feat_cols = [c for c in df.columns if c not in config.EXCLUDE_COLS]
            
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                data_scaled = scaler.transform(df[feat_cols])
            else:
                self.logger.warning(f"⚠️ Scaler not found in {model_dir}. Fitting new scaler (Leakage Risk).")
                scaler = RobustScaler()
                data_scaled = scaler.fit_transform(df[feat_cols])
            
            # -----------------------------------------------------
            # ML Signal 생성
            # -----------------------------------------------------
            ml_model = HybridLearner(model_dir)
            
            # 시퀀스 생성
            X_seq = np.lib.stride_tricks.sliding_window_view(data_scaled, window_shape=(config.ML_SEQ_LEN, len(feat_cols)))
            # 차원 축소 (N, 1, Seq, Feat) -> (N, Seq, Feat)
            if X_seq.ndim == 4:
                X_seq = X_seq.squeeze(axis=1)
                
            X_flat = data_scaled[config.ML_SEQ_LEN:]
            
            # 길이 맞춤
            min_len = min(len(X_seq), len(X_flat))
            
            # 예측
            if not ml_model.load():
                self.logger.error("❌ Failed to load ML models.")
                return -999, 0
                
            signals = ml_model.predict_proba(X_flat[:min_len], X_seq[:min_len])
            
            # DF 슬라이싱 (시퀀스 길이만큼 앞부분 제외)
            sim_df = df.iloc[config.ML_SEQ_LEN:].iloc[:min_len].copy()
            sim_df['ml_signal'] = signals

            # -----------------------------------------------------
            # RL Backtest
            # -----------------------------------------------------
            env = CryptoEnv(sim_df, TradingCore(), precision_df=None, debug=False)
            env = DummyVecEnv([lambda: env])
            
            # VecNormalize 로드
            norm_path = os.path.join(model_dir, "vec_normalize.pkl")
            if os.path.exists(norm_path):
                env = VecNormalize.load(norm_path, env)
                env.training = False
                env.norm_reward = False

            agent_path = os.path.join(model_dir, "final_agent")
            model = PPO.load(agent_path)
            
            obs = env.reset()
            done = [False]
            info = {}
            
            while not done[0]:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, info = env.step(action)
            
            # info가 리스트로 반환됨 (Vectorized Env 특성)
            final_balance = info[0]['final_balance']
            roi = (final_balance - config.INITIAL_BALANCE) / config.INITIAL_BALANCE * 100
            
            return roi, final_balance

        except Exception as e:
            self.logger.error(f"Evaluation Failed for {model_dir}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return -999, 0

    def battle(self, champion_dir, challenger_dir):
        """챔피언 vs 도전자 대결"""
        df = self.load_validation_data()
        
        self.logger.info(f"⚔️ BATTLE START: Champion vs Challenger")
        
        champ_roi, champ_bal = self.evaluate_model(champion_dir, df)
        self.logger.info(f"   🏆 Champion ROI: {champ_roi:.2f}% (${champ_bal:,.2f})")
        
        chall_roi, chall_bal = self.evaluate_model(challenger_dir, df)
        self.logger.info(f"   🥊 Challenger ROI: {chall_roi:.2f}% (${chall_bal:,.2f})")
        
        # 도전자가 5% 이상 더 좋을 때만 승리 (교체 비용 고려)
        if chall_roi > champ_roi * 1.05:
            self.logger.info("🎉 NEW CHAMPION! Challenger Wins.")
            return "challenger"
        else:
            self.logger.info("🛡️ DEFENSE! Champion Remains.")
            return "champion"