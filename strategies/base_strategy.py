from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    모든 매매 전략이 상속받아야 할 기본 클래스
    """
    def __init__(self, config_manager):
        self.cm = config_manager
        self.position = None  # 현재 포지션 상태
        self.history = []     # 매매 기록

    @abstractmethod
    def reset(self):
        """상태 초기화"""
        pass

    @abstractmethod
    def process(self, row, action, balance):
        """
        매 스텝마다 호출되는 핵심 로직
        :param row: 현재 시점 데이터 (Series)
        :param action: 외부 액션
        :param balance: 현재 잔고
        :return: (new_balance, trade_info)
        """
        pass