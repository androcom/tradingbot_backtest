from .base_strategy import BaseStrategy

class DefenseStrategy(BaseStrategy):
    def __init__(self, config_manager):
        super().__init__(config_manager)
        self.conf = self.cm.load_config()
        self.params = self.conf.get('strategy_defense', {})
        self.reset()

    def reset(self):
        self.position = None
        
    def process(self, row, action, balance):
        timestamp = row.name
        trade_occurred = None
        
        # 1. 포지션 관리
        if self.position:
            current_close = row['close']
            entry = self.position['entry_price']
            pnl = (entry - current_close) / entry
            
            triggered = False
            reason = ""
            
            if pnl >= 0.02: 
                triggered = True
                reason = "Defense TP"
            elif pnl <= -0.01: 
                triggered = True
                reason = "Defense SL"
            
            if action == 3:
                triggered = True
                reason = "Manager Stop"
                
            if triggered:
                FEE = 0.0005
                net_pnl = pnl - (FEE * 2)
                balance *= (1 + net_pnl)
                trade_occurred = {
                    'time': timestamp, 'type': 'SHORT',
                    'entry': entry, 'exit': current_close,
                    'pnl': net_pnl, 'balance': balance, 'reason': reason
                }
                self.position = None

        # 2. 신규 진입 (Only Short when signal is strong)
        if self.position is None:
            if action == 2:
                price = row['close']
                SLIPPAGE = 0.0002
                real_entry = price * (1 - SLIPPAGE)
                
                self.position = {
                    'type': 'SHORT',
                    'entry_price': real_entry,
                    'size': 1.0, # RL 호환성
                    'open_time': timestamp
                }
                
        return balance, trade_occurred