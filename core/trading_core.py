import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config

import logging
import pandas as pd
import numpy as np

class TradingCore:
    def __init__(self):
        self.logger = logging.getLogger("TradingCore")
        self.rules = config.TRADING_RULES
        self.reset()

    def reset(self):
        """계좌 및 포지션 상태 초기화"""
        self.balance = config.INITIAL_BALANCE
        self.position = None
        self.history = [] 
        self.trade_count = 0
        self.bankruptcy = False # 파산 여부 플래그

    def get_unrealized_pnl(self, current_price):
        """미실현 손익 계산 (금액 기준)"""
        if not self.position: return 0.0
        entry = self.position['entry_price']
        size = self.position['size']
        
        if self.position['type'] == 'LONG': 
            return (current_price - entry) * size
        else: 
            return (entry - current_price) * size

    def get_unrealized_pnl_pct(self, current_price):
        """미실현 손익률 계산 (%)"""
        if not self.position: return 0.0
        entry = self.position['entry_price']
        pnl = self.get_unrealized_pnl(current_price)
        # 투자 원금(Margin) 대비 수익률
        invested = entry * self.position['size'] / self.position['leverage']
        return (pnl / invested) if invested > 0 else 0.0

    def _calculate_risk_based_size(self, price, sl_price):
        """
        리스크 기반 포지션 사이징 (Kelly Criterion 변형 / 고정 비율 리스크)
        """
        if self.balance <= 0: return 0.0, 0.0
        
        # 1. 리스크 허용 금액 (예: 자산의 1% 손실 허용)
        risk_amount = self.balance * self.rules['risk_per_trade']
        
        # 2. 단위당 예상 손실액 (진입가 - 손절가)
        loss_per_unit = abs(price - sl_price)
        
        # [안전장치] 손절폭이 너무 작으면(0에 수렴) 최소값으로 보정하여 0 나누기 방지
        if loss_per_unit < price * 0.0001:
            loss_per_unit = price * 0.0001
            
        # 3. 리스크에 맞춘 이론적 수량
        ideal_qty = risk_amount / loss_per_unit
        
        # 4. 레버리지 계산 및 제한
        notional_value = ideal_qty * price
        raw_leverage = notional_value / self.balance
        
        # 레버리지 상한선 적용 (Config 설정값)
        final_leverage = min(raw_leverage, config.MAX_LEVERAGE)
        
        # 5. 최종 수량 재계산 (레버리지 제한 반영)
        final_qty = (self.balance * final_leverage) / price
        
        # 최소 주문 금액 미만이면 진입 불가
        if (final_qty * price) < self.rules['min_trade_amount']:
            return 0.0, 0.0
            
        return final_qty, final_leverage

    def process_step(self, action, row_1h, timestamp, precision_candles=None):
        """
        매 스텝(1시간)마다 호출되어 포지션 관리 및 매매 실행
        """
        # 파산 상태면 아무것도 하지 않음
        if self.bankruptcy or self.balance <= 0:
            self.bankruptcy = True
            return

        current_close = row_1h['close']
        current_high = row_1h['high']
        current_low = row_1h['low']
        
        # ATR이나 Trend가 없는 경우 방어 코드
        ema_trend = row_1h.get('ema_trend_4h', row_1h['close'])
        atr = row_1h.get('atr', current_close * 0.01) # 기본값 1%
        
        # 1. 기존 포지션 관리 (Exit, Funding Fee, Trailing Stop)
        if self.position:
            # 펀딩비 차감 (간소화: 1시간마다 발생한다고 가정하거나, 8시간마지만 여기선 매 시간 조금씩 차감)
            position_value = self.position['entry_price'] * self.position['size']
            funding_cost = position_value * self.rules['funding_rate_hourly']
            self.balance -= funding_cost

            # 청산(Liquidation) 체크: 증거금 부족 시 강제 종료
            unrealized_pnl = self.get_unrealized_pnl(current_close)
            equity = self.balance + unrealized_pnl
            
            # 유지 증거금(Maintenance Margin) 대용: 잔고가 0 이하로 떨어지면 파산
            if equity <= 0:
                self._close_position(current_close, timestamp, "LIQUIDATION (Bankruptcy)")
                self.bankruptcy = True
                self.balance = 0
                return

            # 정밀 데이터(5분봉)가 있으면 그것으로 Exit 체크, 없으면 1시간봉 고가/저가로 체크
            exit_triggered = False
            if precision_candles is not None and not precision_candles.empty:
                exit_triggered = self._check_exit_precision(precision_candles, timestamp)
            else:
                exit_triggered = self._check_exit_fallback(current_high, current_low, timestamp)
            
            if exit_triggered: return

        # 2. 신규 진입 로직 (Entry)
        # 트렌드 필터: 정배열(Close > EMA)일 때만 롱, 역배열일 때만 숏
        # (Teacher Tuner 결과에 따라 이 부분은 나중에 On/Off 할 수도 있음)
        trend_up = (current_close > ema_trend)
        trend_down = (current_close < ema_trend)

        if action == 3 and self.position:
            # Action 3: Hold/Exit (포지션이 있으면 청산 신호로 해석 가능, 여기선 Exit Signal로 처리)
            # (전략에 따라 Action 0이 Hold고 3이 Close일 수 있음. RL Env 정의에 따름)
            # 여기서는 RL Env의 Action Space 정의를 따릅니다. (보통 0:Hold, 1:Long, 2:Short, 3:CloseAll)
            self._close_position(current_close, timestamp, "Signal Close")
            
        elif action == 1: # Long Signal
            if self.position:
                if self.position['type'] == 'SHORT':
                    self._close_position(current_close, timestamp, "Switch Long")
                    if trend_up: self._open_position('LONG', current_close, atr, timestamp)
            elif trend_up:
                self._open_position('LONG', current_close, atr, timestamp)
                
        elif action == 2: # Short Signal
            if self.position:
                if self.position['type'] == 'LONG':
                    self._close_position(current_close, timestamp, "Switch Short")
                    if trend_down: self._open_position('SHORT', current_close, atr, timestamp)
            elif trend_down:
                self._open_position('SHORT', current_close, atr, timestamp)

    def _open_position(self, p_type, price, atr, timestamp):
        # 슬리피지 적용
        real_price = price * (1 + config.SLIPPAGE) if p_type == 'LONG' else price * (1 - config.SLIPPAGE)
        
        # 손절가(SL) 계산
        sl_dist = atr * self.rules['sl_atr_multiplier']
        sl_price = real_price - sl_dist if p_type == 'LONG' else real_price + sl_dist
        
        # 사이징 계산
        size, leverage = self._calculate_risk_based_size(real_price, sl_price)
        
        if size <= 0: return

        # 수수료 계산 (진입 시 선차감 방식이 안전함)
        position_value = real_price * size
        fee = position_value * config.FEE_RATE
        
        # 잔고 부족 확인
        if self.balance - fee <= 0:
            return 

        self.balance -= fee
        
        # 포지션 기록
        self.position = {
            'type': p_type,
            'entry_price': real_price,
            'size': size,
            'leverage': leverage,
            'sl': sl_price,
            'base_sl_dist': sl_dist, 
            'highest_price': real_price if p_type == 'LONG' else 0, 
            'lowest_price': real_price if p_type == 'SHORT' else float('inf'),
            'open_time': timestamp
        }

    def _close_position(self, price, timestamp, reason="Close"):
        if not self.position: return
        
        # 슬리피지 적용
        real_price = price * (1 - config.SLIPPAGE) if self.position['type'] == 'LONG' else price * (1 + config.SLIPPAGE)
        
        entry = self.position['entry_price']
        size = self.position['size']
        
        # PnL 계산
        if self.position['type'] == 'LONG': 
            pnl = (real_price - entry) * size
        else: 
            pnl = (entry - real_price) * size
            
        # 수수료 차감
        exit_value = real_price * size
        fee = exit_value * config.FEE_RATE
        
        # 최종 잔고 반영
        net_pnl = pnl - fee
        self.balance += net_pnl
        
        # ROI 계산
        invested = entry * size / self.position['leverage']
        roi = (net_pnl / invested) * 100 if invested > 0 else 0.0
        
        # 기록 저장
        self.history.append({
            'open_time': self.position['open_time'],
            'close_time': timestamp,
            'type': self.position['type'],
            'entry_price': entry,
            'exit_price': real_price,
            'leverage': self.position['leverage'],
            'size': size,
            'pnl': net_pnl,
            'roi': roi,
            'reason': reason,
            'balance': self.balance
        })
        
        self.trade_count += 1
        self.position = None

    def _check_exit_precision(self, candles_5m, timestamp_1h):
        """5분봉 데이터를 이용한 정밀 청산 (SL/TP/Trailing)"""
        if not self.position: return False
        
        p_type = self.position['type']
        
        for ts_5m, row in candles_5m.iterrows():
            curr_high, curr_low = row['high'], row['low']
            
            # 1. Stop Loss 체크
            if (p_type == 'LONG' and curr_low <= self.position['sl']) or \
               (p_type == 'SHORT' and curr_high >= self.position['sl']):
                self._close_position(self.position['sl'], ts_5m, "StopLoss")
                return True
                
            # 2. Trailing Stop 업데이트
            self._update_stops(curr_high, curr_low, self.position['entry_price'])
            
        return False

    def _check_exit_fallback(self, high, low, timestamp):
        """1시간봉 고가/저가를 이용한 청산 (5분봉 없을 때)"""
        if not self.position: return False
        
        # 1. Stop Loss 체크 (보수적으로 먼저 맞았다고 가정)
        if self.position['type'] == 'LONG':
            if low <= self.position['sl']:
                self._close_position(self.position['sl'], timestamp, "StopLoss(Fallback)")
                return True
            self._update_stops(high, low, self.position['entry_price'])
            
        else: # SHORT
            if high >= self.position['sl']:
                self._close_position(self.position['sl'], timestamp, "StopLoss(Fallback)")
                return True
            self._update_stops(high, low, self.position['entry_price'])
            
        return False

    def _update_stops(self, curr_high, curr_low, entry_price):
        """Trailing Stop 로직"""
        dist = self.position['base_sl_dist']
        
        # ATR Multiplier 역산 (현재 dist 기준)
        # (원래 ATR을 저장했어야 하나, 메모리 절약을 위해 역산하거나 저장된 dist 사용)
        # 여기서는 단순히 최초 SL 거리를 기준으로 비율 계산
        
        trigger_ratio = self.rules['tp_trigger_atr'] # 예: 2.0 ATR 수익 시
        gap_ratio = self.rules['trailing_gap_atr']   # 예: 1.0 ATR 간격 유지
        
        # ATR을 정확히 모르므로, 초기 SL 거리(sl_dist)가 'sl_atr_multiplier'배 였음을 이용
        # 1 ATR = dist / sl_atr_multiplier
        one_atr = dist / self.rules['sl_atr_multiplier']
        
        trigger_dist = one_atr * trigger_ratio
        gap_dist = one_atr * gap_ratio
        
        if self.position['type'] == 'LONG':
            # 고점 갱신
            if curr_high > self.position['highest_price']:
                self.position['highest_price'] = curr_high
                
                # 발동 조건: 진입가보다 일정 수준 이상 올랐을 때
                if curr_high > entry_price + trigger_dist:
                    # 새로운 SL = 현재 고점 - 갭
                    new_sl = curr_high - gap_dist
                    # SL은 위로만 움직임 (손실 줄이기/수익 확정)
                    # 단, 최소 본전(Break Even) 이상으로 올릴 때
                    if new_sl > self.position['sl']:
                        self.position['sl'] = new_sl

        else: # SHORT
            # 저점 갱신
            if curr_low < self.position['lowest_price']:
                self.position['lowest_price'] = curr_low
                
                if curr_low < entry_price - trigger_dist:
                    new_sl = curr_low + gap_dist
                    # SL은 아래로만 움직임
                    if new_sl < self.position['sl']:
                        self.position['sl'] = new_sl