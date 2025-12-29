import logging
import sys
import os

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import constants as config
from core.config_manager import ConfigManager
from strategies.trend_strategy import TrendStrategy
from strategies.range_strategy import RangeStrategy
from strategies.defense_strategy import DefenseStrategy

class TradingCore:
    def __init__(self, strategy_name='strategy_trend'):
        """
        :param strategy_name: 사용할 전략 이름 ('strategy_trend', 'strategy_range', 'strategy_defense')
        """
        self.logger = logging.getLogger("TradingCore")
        self.cm = ConfigManager(config.MAIN_SYMBOL)
        self.strategy_name = strategy_name
        
        # [Strategy Factory] 전략 선택 로직
        if strategy_name == 'strategy_trend':
            self.strategy = TrendStrategy(self.cm)
        elif strategy_name == 'strategy_range':
            self.strategy = RangeStrategy(self.cm)
        elif strategy_name == 'strategy_defense':
            self.strategy = DefenseStrategy(self.cm)
        else:
            self.logger.warning(f"Unknown strategy {strategy_name}, falling back to TrendStrategy.")
            self.strategy = TrendStrategy(self.cm)
            
        self.balance = config.INITIAL_BALANCE
        self.history = []
        self.reset()

    def reset(self):
        """계좌 및 전략 초기화"""
        self.balance = config.INITIAL_BALANCE
        self.history = []
        self.strategy.reset()

    @property
    def position(self):
        """
        RL Env가 position 상태를 조회할 때 사용 (전략 내부 변수 전달)
        """
        return self.strategy.position

    def get_unrealized_pnl(self, current_price):
        """
        RL Env 관측용: 현재 미실현 손익률 계산
        """
        pos = self.strategy.position
        if not pos: return 0.0
        
        entry = pos['entry_price']
        if pos['type'] == 'LONG':
            return (current_price - entry) / entry
        else:
            return (entry - current_price) / entry
            
    def get_unrealized_pnl_pct(self, current_price):
        """호환성 유지용"""
        return self.get_unrealized_pnl(current_price)

    def process_step(self, action, row, timestamp, precision_candles=None):
        """
        RL Env의 step()에서 호출되는 메인 함수.
        실제 매매 판단과 계산은 self.strategy.process()로 위임합니다.
        
        :param action: RL/Model의 Action (0:Hold, 1:Long, 2:Short, 3:Close 등)
        :param row: 현재 캔들 데이터
        :param timestamp: 현재 시간
        :param precision_candles: (옵션) 정밀 시뮬레이션용 5분봉 데이터
        """
        # 전략 실행
        # new_balance: 업데이트된 잔고
        # trade_info: 체결 내역 (체결 없으면 None)
        new_balance, trade_info = self.strategy.process(row, action, self.balance)
        
        # 잔고 업데이트
        self.balance = new_balance
        
        # 거래 기록이 있으면 저장
        if trade_info:
            trade_info['balance'] = self.balance
            self.history.append(trade_info)