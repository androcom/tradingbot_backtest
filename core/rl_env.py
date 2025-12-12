import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque

class CryptoEnv(gym.Env):
    def __init__(self, df, trading_logic, precision_df=None, debug=False):
        super(CryptoEnv, self).__init__()
        self.df = df
        self.precision_df = precision_df
        self.logic = trading_logic
        self.debug = debug
        
        self.features = [c for c in df.columns if c not in config.EXCLUDE_COLS and 'ml_signal' not in c]
        self.action_space = spaces.Discrete(4)
        
        # Obs: Market Features + Position State (Size, PriceRatio, PnL, ML_Signal)
        obs_dim = len(self.features) + 4 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.returns_buffer = deque(maxlen=30)
        self.max_equity = config.INITIAL_BALANCE
        
        # Config의 파라미터 로드
        if hasattr(config, 'REWARD_PARAMS'):
            self.reward_params = config.REWARD_PARAMS
        else:
            self.reward_params = {
                'profit_scale': 200.0,
                'teacher_bonus': 0.05,
                'teacher_penalty': 0.1,
                'mdd_penalty_factor': 1.0,
                'new_high_bonus': 0.5
            }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.logic.reset()
        self.returns_buffer.clear()
        self.max_equity = config.INITIAL_BALANCE
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        market_obs = row[self.features].values.astype(np.float32)
        
        if self.logic.position:
            pos_size = float(self.logic.position['size'])
            entry_price = float(self.logic.position['entry_price'])
            price_ratio = (row['close'] / entry_price) if entry_price > 0 else 1.0
            pnl_pct = self.logic.get_unrealized_pnl_pct(row['close'])
        else:
            pos_size = 0.0
            price_ratio = 1.0
            pnl_pct = 0.0
            
        ml_sig = row.get('ml_signal', 0.0)
        pos_obs = np.array([pos_size, price_ratio, pnl_pct, ml_sig], dtype=np.float32)
        
        return np.concatenate([market_obs, pos_obs])

    def step(self, action):
        row = self.df.iloc[self.current_step]
        current_ts = row.name
        current_close = row['close']
        ml_signal = row.get('ml_signal', 0.0)
        
        # Step 전 자산 (Unrealized PnL 포함)
        prev_equity = self.logic.balance + self.logic.get_unrealized_pnl(current_close)
        
        # 정밀 데이터(5분봉) 로드 (청산 시뮬레이션용)
        precision_candles = None
        if self.precision_df is not None:
            try:
                end_ts = current_ts + pd.Timedelta(minutes=59)
                precision_candles = self.precision_df.loc[current_ts:end_ts]
            except KeyError: pass 

        # 트레이딩 로직 실행
        self.logic.process_step(action, row, current_ts, precision_candles)
        
        # Step 후 자산
        curr_equity = self.logic.balance + self.logic.get_unrealized_pnl(current_close)
        
        # 보상 계산
        reward = self._calculate_reward(prev_equity, curr_equity, action, ml_signal)
        
        self.current_step += 1
        terminated = False
        truncated = False
        info = {}
        
        # 종료 조건 1: 데이터 끝
        if self.current_step >= len(self.df) - 1:
            terminated = True
            info['final_balance'] = curr_equity
            
        # 종료 조건 2: 파산 (자산이 40% 미만으로 감소)
        if curr_equity < config.INITIAL_BALANCE * 0.4:
            terminated = True
            reward = -10.0 # 큰 페널티
            info['final_balance'] = curr_equity
        
        return self._get_obs(), reward, terminated, truncated, info

    def _calculate_reward(self, prev_equity, curr_equity, action, ml_signal):
        p = self.reward_params
        
        # [수정] 로그 수익률 계산 시 안전장치 (0 나누기 방지)
        epsilon = 1e-8
        safe_prev = max(prev_equity, epsilon)
        safe_curr = max(curr_equity, epsilon)
        
        log_return = np.log(safe_curr / safe_prev)
        step_reward = log_return * p['profit_scale']
        
        # Teacher Signal 보너스/페널티
        # ML 신호가 강할 때 (>0.5 or <-0.5) 반대로 가면 페널티
        if ml_signal > 0.5: # Strong Buy Signal
            if action == 1: step_reward += p['teacher_bonus'] # Long
            elif action == 2: step_reward -= p['teacher_penalty'] # Short
        elif ml_signal < -0.5: # Strong Sell Signal
            if action == 2: step_reward += p['teacher_bonus'] # Short
            elif action == 1: step_reward -= p['teacher_penalty'] # Long
            
        # 신고가 갱신 보너스
        if curr_equity > self.max_equity:
            self.max_equity = curr_equity
            step_reward += p['new_high_bonus']
        
        # Drawdown 페널티 (최고점 대비 하락폭)
        if self.max_equity > 0:
            drawdown = (self.max_equity - curr_equity) / self.max_equity
            if drawdown > 0.1: # 10% 이상 하락 시
                step_reward -= (drawdown * p['mdd_penalty_factor'])

        # 보상 클리핑 (학습 안정성)
        return np.clip(step_reward, -5.0, 5.0)