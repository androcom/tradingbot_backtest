# rl_strategy/rl_env.py
import sys
import os

# 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import config # 루트 모듈
from trading_engine import AccountManager # 루트 모듈

class CryptoTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, leverage=1):
        super(CryptoTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.total_steps = len(self.df)
        self.current_step = 0
        self.initial_balance = initial_balance
        self.leverage = leverage
        
        # Action: 0=Hold, 1=Long, 2=Short, 3=Close
        self.action_space = spaces.Discrete(4)
        
        # 제외할 컬럼 (가격 정보 등은 직접 피처로 쓰지 않고 가공된 지표 사용)
        exclude_cols = ['timestamp', 'target_cls', 'open', 'high', 'low', 'close', 'volume']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # [확인] ml_signal이 포함되었는지 체크
        if 'ml_signal' in self.feature_cols:
            # print(f"   [Env] 'ml_signal' detected! utilizing teacher knowledge.")
            pass
            
        # Obs: Features + Position Info (3)
        n_features = len(self.feature_cols) + 3 
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        
        self.account = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.account = AccountManager(balance=self.initial_balance, leverage=self.leverage)
        return self._next_observation(), {}

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        market_obs = row[self.feature_cols].values.astype(np.float32)
        
        pos_size = self.account.position['size'] if self.account.position else 0.0
        entry_price = self.account.position['price'] if self.account.position else 0.0
        
        curr_price = row['close'] # close 가격은 df에 남아있음 (피처로는 안쓰지만)
        pnl_pct = 0.0
        if self.account.position:
            if self.account.position['type'] == 'LONG':
                pnl_pct = (curr_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - curr_price) / entry_price
        
        obs = np.concatenate((market_obs, [pos_size, entry_price, pnl_pct]))
        return obs.astype(np.float32)

    def step(self, action):
        prev_balance = self.account.balance
        
        # 현재 step의 데이터
        current_row = self.df.iloc[self.current_step]
        current_price = current_row['open'] 
        timestamp = current_row.name # 인덱스가 리셋되었으므로 의미 없을 수 있음, 디버그용
        
        # Trading Action
        signal = 'HOLD'
        if action == 1: signal = 'OPEN_LONG'
        elif action == 2: signal = 'OPEN_SHORT'
        elif action == 3: signal = 'CLOSE' # Close 신호 명시적 처리
        
        # 매매 실행 (Trading Engine 위임)
        # RL 학습 초기에는 과감한 투자를 위해 비중을 높임 (0.99)
        qty = 0
        if signal in ['OPEN_LONG', 'OPEN_SHORT']:
             qty = (self.account.balance * 0.99 * self.leverage) / current_price
             
        # 기존 Trading Engine 활용
        # 주의: execute_trade는 'CLOSE' 신호를 직접 받지 않고 반대 포지션 오픈 시 청산함.
        # 따라서 Action 3(Close)일 때는 강제 청산 로직 호출
        if signal == 'CLOSE' and self.account.position:
            self.account._force_close(current_price, timestamp, "RL_Action")
        else:
            self.account.execute_trade(signal, current_price, qty, self.leverage, timestamp)
        
        # Step 진행
        self.current_step += 1
        terminated = (self.current_step >= self.total_steps - 1)
        truncated = False
        
        # 보상 계산 (Reward Engineering)
        equity = self.account.balance
        
        # 포지션 평가액 합산
        if self.account.position:
            p_size = self.account.position['size']
            p_entry = self.account.position['price']
            # 다음 봉의 종가가 아닌 현재 봉의 종가(또는 다음 봉 시가)로 평가
            # 여기서는 보수적으로 current_step의 close 사용
            curr_c = self.df.iloc[self.current_step]['close']
            
            if self.account.position['type'] == 'LONG':
                unrealized_pnl = (curr_c - p_entry) * p_size
            else:
                unrealized_pnl = (p_entry - curr_c) * p_size
            equity += unrealized_pnl
            
        # 1. 수익률 보상
        step_reward = (equity - prev_balance) / prev_balance * 100
        
        # 2. 해설지(Teacher) 일치 보너스 (선택 사항 - 여기서는 간접 학습 유도)
        # ml_signal이 강한데 반대로 가면 패널티를 줄 수도 있음.
        # 하지만 이미 ml_signal이 observation에 있으므로 PPO가 알아서 학습할 것임.
        
        reward = step_reward
        
        # 파산 시 큰 패널티
        if equity < self.initial_balance * 0.5:
            reward = -100
            terminated = True
            
        return self._next_observation(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        pass