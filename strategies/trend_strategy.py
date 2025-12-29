import numpy as np
from .base_strategy import BaseStrategy

class TrendStrategy(BaseStrategy):
    def __init__(self, config_manager):
        super().__init__(config_manager)
        # ConfigManager를 통해 설정 로드
        self.conf = self.cm.load_config()
        self.params = self.conf.get('strategy_trend', {}).get('trading_rules', {})
        self.reset()

    def reset(self):
        self.position = None 
        self.highest_price = 0.0
        self.lowest_price = 0.0
        self.trade_count = 0
        
    def process(self, row, action, balance):
        current_close = row['close']
        current_high = row['high']
        current_low = row['low']
        timestamp = row.name 
        
        # 파라미터 로드 (기본값 설정)
        tp_rate = self.params.get('tp_rate', 0.05)
        sl_rate = self.params.get('sl_rate', 0.02)
        trailing_gap = self.params.get('trailing_gap_rate', tp_rate * 0.5)
        
        FEE = 0.0005
        SLIPPAGE = 0.0002
        
        trade_occurred = None
        
        # 1. 포지션 관리
        if self.position:
            pnl_pct = 0.0
            triggered = False
            reason = ""
            entry = self.position['entry_price']
            
            if self.position['type'] == 'LONG':
                if current_high > self.highest_price:
                    self.highest_price = current_high
                    if self.highest_price > entry * (1 + tp_rate):
                        new_sl = self.highest_price * (1 - trailing_gap)
                        if new_sl > self.position['sl']:
                            self.position['sl'] = new_sl
                            
                if current_low <= self.position['sl']:
                    exit_price = self.position['sl']
                    pnl_pct = (exit_price - entry) / entry
                    triggered = True
                    reason = "Trailing/SL"

            elif self.position['type'] == 'SHORT':
                if current_low < self.lowest_price:
                    self.lowest_price = current_low
                    if self.lowest_price < entry * (1 - tp_rate):
                        new_sl = self.lowest_price * (1 + trailing_gap)
                        if new_sl < self.position['sl']:
                            self.position['sl'] = new_sl
                            
                if current_high >= self.position['sl']:
                    exit_price = self.position['sl']
                    pnl_pct = (entry - exit_price) / entry
                    triggered = True
                    reason = "Trailing/SL"
            
            if action == 3 and not triggered:
                exit_price = current_close
                if self.position['type'] == 'LONG': pnl_pct = (exit_price - entry) / entry
                else: pnl_pct = (entry - exit_price) / entry
                triggered = True
                reason = "Signal Close"

            if triggered:
                net_pnl = pnl_pct - (FEE * 2)
                balance *= (1 + net_pnl)
                trade_occurred = {
                    'time': timestamp,
                    'type': self.position['type'],
                    'entry': entry,
                    'exit': exit_price,
                    'pnl': net_pnl,
                    'balance': balance,
                    'reason': reason
                }
                self.position = None

        # 2. 신규 진입
        if self.position is None:
            entry_type = None
            if action == 1: entry_type = 'LONG'
            elif action == 2: entry_type = 'SHORT'
            
            if entry_type:
                price = current_close
                real_entry = price * (1 + SLIPPAGE) if entry_type == 'LONG' else price * (1 - SLIPPAGE)
                
                if entry_type == 'LONG':
                    sl_price = real_entry * (1 - sl_rate)
                    self.highest_price = real_entry
                else:
                    sl_price = real_entry * (1 + sl_rate)
                    self.lowest_price = real_entry
                
                # [FIX] 'size': 1.0 추가 (RL 호환성)
                self.position = {
                    'type': entry_type,
                    'entry_price': real_entry,
                    'size': 1.0, 
                    'sl': sl_price,
                    'open_time': timestamp
                }
                self.trade_count += 1

        return balance, trade_occurred