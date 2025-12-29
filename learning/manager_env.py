import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from core import constants as config
from execution.trade_engine import TradingCore

class RLManagerEnv(gym.Env):
    """
    [상위 관리자 환경]
    - State: 시장 국면 지표 (ADX, RSI, BB_Width, ATR_Ratio, Close_vs_EMA)
    - Action: 0(Trend), 1(Range), 2(Defense)
    - Reward: 포트폴리오 가치 변동분
    """
    def __init__(self, df, debug=False):
        super(RLManagerEnv, self).__init__()
        self.df = df
        self.debug = debug
        
        # 3가지 전략 엔진 대기
        self.engines = {
            0: TradingCore('strategy_trend'),
            1: TradingCore('strategy_range'),
            2: TradingCore('strategy_defense')
        }
        
        # Action: 3개 전략 중 하나 선택
        self.action_space = spaces.Discrete(3)
        
        # Obs: [ADX, RSI, BB_Width, EMA_Ratio, ATR_Ratio]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
        self.balance = config.INITIAL_BALANCE
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = config.INITIAL_BALANCE
        
        for engine in self.engines.values():
            engine.reset()
            engine.balance = self.balance
            
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        
        adx = row.get('adx', 0)
        rsi = row.get('rsi', 50)
        bb_w = row.get('bb_w', 0)
        ema_ratio = (row['close'] - row.get('ema_trend', row['close'])) / row['close']
        atr_ratio = row.get('atr', 0) / row['close']
        
        obs = np.array([adx, rsi, bb_w, ema_ratio, atr_ratio], dtype=np.float32)
        return np.nan_to_num(obs)

    def step(self, action):
        active_engine = self.engines[action]
        active_engine.balance = self.balance # 잔고 동기화
        
        row = self.df.iloc[self.current_step]
        timestamp = row.name
        
        # 하위 전략 실행 (Action은 전략 내부 로직에 의해 무시되거나 참고됨)
        # TrendStrategy는 ml_signal이 필요하므로 DataFrame에 이미 주입되어 있어야 함
        active_engine.process_step(0, row, timestamp) 
        
        prev_balance = self.balance
        self.balance = active_engine.balance
        
        # 나머지 엔진 잔고 동기화
        for k, engine in self.engines.items():
            engine.balance = self.balance

        # 보상: 자산 증감률
        reward = (self.balance - prev_balance) / prev_balance if prev_balance > 0 else 0
        
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        info = {
            'balance': self.balance,
            'active_strategy': action
        }
        
        return self._get_obs(), reward, terminated, truncated, info