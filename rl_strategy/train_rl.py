# rl_strategy/train_rl.py
import pandas as pd
import numpy as np
import os
import torch
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor 
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.preprocessing import RobustScaler

# 경로 설정 (sys.path 추가)
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import config
from data_processor import DataProcessor
from rl_env import CryptoTradingEnv
from ml_strategy.ai_models import HybridEnsemble

TENSORBOARD_LOG_DIR = os.path.join(root_dir, "logs", "tensorboard")
if not os.path.exists(TENSORBOARD_LOG_DIR):
    os.makedirs(TENSORBOARD_LOG_DIR)

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
    def _on_step(self) -> bool:
        return True

def make_env(rank, df, seed=0):
    def _init():
        env = CryptoTradingEnv(df)
        env = Monitor(env) 
        env.reset(seed=seed + rank)
        return env
    return _init

def train_ml_and_generate_signals(symbol, train_df, test_df, features):
    """
    [Real-World Simulation]
    1. Train Set으로만 ML 모델을 학습시킵니다.
    2. Train Set에 대한 예측값(ml_signal)을 생성합니다.
    3. Test Set에 대한 예측값(ml_signal)을 생성합니다. (미래 정보 미포함)
    """
    print(f"\n>> [Teacher] Training ML Model on TRAIN DATA ONLY (No Leakage)...")
    
    # 1. 데이터 준비 (LSTM 시퀀스 + XGB 플랫)
    dp = DataProcessor(symbol)
    
    # Train Data 준비
    X_train_seq, y_train_seq = dp.create_sequences(train_df, features, config.LSTM_WINDOW)
    # LSTM 윈도우 크기만큼 앞부분은 학습 데이터에서 제외됨
    X_train_flat = train_df.iloc[config.LSTM_WINDOW:][features].values
    y_train_flat = train_df.iloc[config.LSTM_WINDOW:]['target_cls'].values # 학습용 타겟

    # 2. 모델 학습 (Train Data만 사용!)
    ml_model = HybridEnsemble(symbol)
    # y_train_flat은 타겟 클래스(0,1,2)가 있어야 함
    # (참고: RL 학습 전처리 단계에서 target_cls가 보존되어 있어야 함)
    ml_model.train(X_train_seq, y_train_seq, X_train_flat, y_train_flat)
    
    print(f">> [Teacher] Generating signals...")

    # 3. Train Set 예측 (In-Sample Score)
    train_scores = ml_model.batch_predict(X_train_seq, X_train_flat)
    
    # 4. Test Set 예측 (Out-of-Sample Score -> Real Simulation)
    X_test_seq, _ = dp.create_sequences(test_df, features, config.LSTM_WINDOW)
    X_test_flat = test_df.iloc[config.LSTM_WINDOW:][features].values
    
    test_scores = ml_model.batch_predict(X_test_seq, X_test_flat)
    
    # 5. 데이터프레임에 적용 (윈도우만큼 잘린 부분 고려하여 복사본 생성)
    # Train DF 처리
    train_df_trimmed = train_df.iloc[config.LSTM_WINDOW:].copy()
    train_df_trimmed['ml_signal'] = train_scores
    
    # Test DF 처리
    test_df_trimmed = test_df.iloc[config.LSTM_WINDOW:].copy()
    test_df_trimmed['ml_signal'] = test_scores
    
    print(f"   - Train Signals: {len(train_scores)}, Test Signals: {len(test_scores)}")
    
    return train_df_trimmed, test_df_trimmed

def train_and_test():
    symbol = 'BTC/USDT'
    print(f"\n{'='*60}")
    print(f">> RL TRAINING (STRICT NO-LEAKAGE MODE): {symbol}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">> Device Check: {device.upper()}")

    # 1. Raw 데이터 로드
    print(">> Loading Raw Data...")
    dp = DataProcessor(symbol)
    full_df = dp.prepare_multi_timeframe_data()
    
    # 2. 데이터 분할 (Split FIRST)
    # 미래 데이터를 절대 참조하지 않도록 먼저 자릅니다.
    split_idx = int(len(full_df) * 0.8)
    raw_train_df = full_df.iloc[:split_idx].copy()
    raw_test_df = full_df.iloc[split_idx:].copy()
    
    print(f">> Split Data: Train({len(raw_train_df)}) / Test({len(raw_test_df)})")

    # 3. 스케일링 (Fit on Train, Transform on Test)
    exclude_cols = ['timestamp', 'target_cls', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [c for c in full_df.columns if c not in exclude_cols]
    
    print(">> Scaling Features (Fit on Train ONLY)...")
    scaler = RobustScaler()
    
    # Train Set으로 학습
    raw_train_df[feature_cols] = scaler.fit_transform(raw_train_df[feature_cols])
    # Test Set은 변환만 (미래 통계 정보 미포함)
    raw_test_df[feature_cols] = scaler.transform(raw_test_df[feature_cols])
    
    # 4. ML 해설지 생성 (Strict Mode)
    # 여기서 ML 모델도 Train Set으로만 학습하고, Test Set을 예측합니다.
    train_df, test_df = train_ml_and_generate_signals(symbol, raw_train_df, raw_test_df, feature_cols)
    
    # 인덱스 리셋 (환경 구동용)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # 5. 벡터화 환경 생성
    num_cpu = multiprocessing.cpu_count()
    n_envs = max(1, num_cpu - 2) 
    print(f">> Creating {n_envs} parallel environments...")
    
    train_env = SubprocVecEnv([make_env(i, train_df) for i in range(n_envs)])
    
    # 6. PPO 모델 정의
    print(">> Initializing PPO Agent...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03, 
        device=device,
        tensorboard_log=TENSORBOARD_LOG_DIR
    )

    # 7. 학습 시작
    total_timesteps = 2_000_000 
    print(f">> Starting Training for {total_timesteps} timesteps...")
    
    model.learn(
        total_timesteps=total_timesteps, 
        progress_bar=True,
        callback=TensorboardCallback()
    )

    # 모델 저장
    save_path = os.path.join(config.MODEL_DIR, "ppo_btc_strict_agent")
    model.save(save_path)
    print(f">> Model saved to {save_path}.zip")

    # 8. 백테스트 (Validation)
    print(f"\n{'='*60}")
    print(">> Starting REALISTIC BACKTEST on Test Set...")
    print(f"{'='*60}")
    
    test_env = CryptoTradingEnv(test_df) 
    obs, _ = test_env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        
    final_balance = test_env.account.balance
    initial_balance = test_env.initial_balance
    roi = ((final_balance - initial_balance) / initial_balance) * 100
    
    print(f"Initial Balance : ${initial_balance:,.2f}")
    print(f"Final Balance   : ${final_balance:,.2f}")
    print(f"Total ROI       : {roi:+.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    train_and_test()