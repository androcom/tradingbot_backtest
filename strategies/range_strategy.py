from .base_strategy import BaseStrategy

class RangeStrategy(BaseStrategy):
    def __init__(self, config_manager):
        super().__init__(config_manager)
        self.conf = self.cm.load_config()
        self.params = self.conf.get('strategy_range', {})
        self.reset()

    def reset(self):
        self.position = None
        self.trade_count = 0

    def process(self, row, action, balance):
        current_close = row['close']
        current_rsi = row.get('rsi', 50)
        timestamp = row.name
        
        tp_rate = self.params.get('tp_rate', 0.01)
        sl_rate = self.params.get('sl_rate', 0.02)
        rsi_lower = self.params.get('rsi_lower', 30)
        rsi_upper = self.params.get('rsi_upper', 70)
        
        FEE = 0.0005
        SLIPPAGE = 0.0002
        trade_occurred = None

        # 1. 포지션 관리 (Fixed TP/SL)
        if self.position:
            entry = self.position['entry_price']
            pnl_pct = 0.0
            triggered = False
            exit_price = 0.0
            reason = ""

            if self.position['type'] == 'LONG':
                if current_close >= entry * (1 + tp_rate):
                    exit_price = entry * (1 + tp_rate)
                    pnl_pct = tp_rate
                    triggered = True
                    reason = "Fixed TP"
                elif current_close <= entry * (1 - sl_rate):
                    exit_price = entry * (1 - sl_rate)
                    pnl_pct = -sl_rate
                    triggered = True
                    reason = "Fixed SL"

            elif self.position['type'] == 'SHORT':
                if current_close <= entry * (1 - tp_rate):
                    exit_price = entry * (1 - tp_rate)
                    pnl_pct = tp_rate
                    triggered = True
                    reason = "Fixed TP"
                elif current_close >= entry * (1 + sl_rate):
                    exit_price = entry * (1 + sl_rate)
                    pnl_pct = -sl_rate
                    triggered = True
                    reason = "Fixed SL"

            if action == 3 and not triggered:
                exit_price = current_close
                if self.position['type'] == 'LONG': pnl_pct = (exit_price - entry)/entry
                else: pnl_pct = (entry - exit_price)/entry
                triggered = True
                reason = "Signal Close"

            if triggered:
                net_pnl = pnl_pct - (FEE * 2)
                balance *= (1 + net_pnl)
                trade_occurred = {
                    'time': timestamp, 'type': self.position['type'],
                    'entry': entry, 'exit': exit_price,
                    'pnl': net_pnl, 'balance': balance, 'reason': reason
                }
                self.position = None

        # 2. 신규 진입 (RSI)
        if self.position is None:
            entry_type = None
            if current_rsi < rsi_lower: entry_type = 'LONG'
            elif current_rsi > rsi_upper: entry_type = 'SHORT'
            
            if entry_type:
                price = current_close
                real_entry = price * (1 + SLIPPAGE) if entry_type == 'LONG' else price * (1 - SLIPPAGE)
                
                self.position = {
                    'type': entry_type,
                    'entry_price': real_entry,
                    'size': 1.0, # RL 호환성
                    'open_time': timestamp
                }
                self.trade_count += 1
                
        return balance, trade_occurred