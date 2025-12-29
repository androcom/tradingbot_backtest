import sys
import os
import pandas as pd
import numpy as np
import ccxt
import ta
import time
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import constants as config

class DataProcessor:
    def __init__(self, logger=None):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': { 'defaultType': 'future' }
        })
        self.logger = logger if logger else logging.getLogger("DataProcessor")

    def log(self, msg):
        self.logger.info(msg)

    def _download_range(self, symbol, timeframe, start_ts, end_ts):
        all_ohlcv = []
        since = start_ts
        limit = 1000 
        
        current_server_time = self.exchange.milliseconds()
        if end_ts > current_server_time:
            end_ts = current_server_time

        while since < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit=limit)
                if not ohlcv: break
                last_fetched_ts = ohlcv[-1][0]
                if last_fetched_ts == since: break 
                since = last_fetched_ts + 1
                all_ohlcv.extend(ohlcv)
                time.sleep(self.exchange.rateLimit / 1000)
            except Exception as e:
                self.log(f"[Data] Error during download: {e}")
                time.sleep(5)
                continue
                
        if not all_ohlcv: return pd.DataFrame()
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        return df

    def fetch_data(self, symbol, timeframe, start_str, end_str):
        safe_symbol = symbol.replace('/', '_')
        file_path = os.path.join(config.DATA_DIR, f"{safe_symbol}_{timeframe}_future.parquet")
        
        req_start_dt = pd.to_datetime(start_str)
        req_end_dt = pd.to_datetime(end_str)
        req_start_ts = self.exchange.parse8601(start_str)
        req_end_ts = self.exchange.parse8601(end_str)

        df = pd.DataFrame()
        file_exists = False

        if os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path)
                df = df[~df.index.duplicated(keep='last')].sort_index()
                if not df.empty: file_exists = True
            except Exception as e:
                self.log(f"[Data] Corrupted file, re-downloading: {e}")
                df = pd.DataFrame()

        dfs_to_concat = []
        is_updated = False

        if not file_exists:
            self.log(f"[Data] Downloading {symbol} ({timeframe}) FUTURES data...")
            new_df = self._download_range(symbol, timeframe, req_start_ts, req_end_ts)
            if not new_df.empty:
                dfs_to_concat.append(new_df)
                is_updated = True
        else:
            local_start_dt = df.index[0]
            local_end_dt = df.index[-1]
            local_start_ts = int(local_start_dt.timestamp() * 1000)
            local_end_ts = int(local_end_dt.timestamp() * 1000)

            if req_start_dt < local_start_dt - pd.Timedelta(hours=1):
                self.log(f"[Data] Fetching Missing Head...")
                head_df = self._download_range(symbol, timeframe, req_start_ts, local_start_ts)
                if not head_df.empty:
                    dfs_to_concat.append(head_df)
                    is_updated = True
            dfs_to_concat.append(df)
            if req_end_dt > local_end_dt + pd.Timedelta(hours=1):
                self.log(f"[Data] Fetching Missing Tail...")
                tail_df = self._download_range(symbol, timeframe, local_end_ts + 1, req_end_ts)
                if not tail_df.empty:
                    dfs_to_concat.append(tail_df)
                    is_updated = True

        if is_updated and dfs_to_concat:
            full_df = pd.concat(dfs_to_concat)
            full_df = full_df[~full_df.index.duplicated(keep='last')].sort_index()
            full_df.to_parquet(file_path)
            self.log(f"[Data] Saved to {file_path} (Rows: {len(full_df)})")
            df = full_df
        
        if df.empty: return df
        mask = (df.index >= req_start_dt) & (df.index <= req_end_dt)
        return df.loc[mask]

    def _add_technical_indicators(self, df, window, suffix=''):
        if df.empty: return df
        df = df.copy()
        
        # Trend
        trend_win = config.TRADING_RULES['trend_window']
        ema = ta.trend.EMAIndicator(df['close'], window=trend_win)
        df[f'ema_trend{suffix}'] = ema.ema_indicator()
        
        # ADX (Manager용)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df[f'adx{suffix}'] = adx.adx()
        
        # Momentum
        df[f'rsi{suffix}'] = ta.momentum.RSIIndicator(df['close'], window=window).rsi()
        
        # Volatility
        df[f'atr{suffix}'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=window).average_true_range()
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df[f'bb_w{suffix}'] = bb.bollinger_wband()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df[f'macd{suffix}'] = macd.macd()
        return df

    def add_indicators(self, df, window=config.INDICATOR_WINDOW):
        df = self._add_technical_indicators(df, window, suffix='')
        df.dropna(inplace=True)
        return df

    def get_ml_data(self, symbol=config.MAIN_SYMBOL):
        df_main = self.fetch_data(symbol, config.TIMEFRAME_MAIN, config.DATE_START, config.DATE_END)
        if df_main.empty: return df_main
        df_main = self._add_technical_indicators(df_main, config.INDICATOR_WINDOW, suffix='')
        
        df_aux = self.fetch_data(symbol, config.TIMEFRAME_AUX, config.DATE_START, config.DATE_END)
        if not df_aux.empty:
            df_aux = self._add_technical_indicators(df_aux, config.INDICATOR_WINDOW, suffix='_4h')
            df_aux_resampled = df_aux.resample(config.TIMEFRAME_MAIN).ffill()
            aux_cols = [c for c in df_aux_resampled.columns if '_4h' in c]
            df_aux_features = df_aux_resampled[aux_cols]
            try:
                lag = int(pd.Timedelta(config.TIMEFRAME_AUX) / pd.Timedelta(config.TIMEFRAME_MAIN))
                df_aux_features = df_aux_features.shift(lag)
            except:
                df_aux_features = df_aux_features.shift(1)
            df_main = df_main.join(df_aux_features)
            df_main.dropna(inplace=True)

        df_main = self.create_target(df_main)
        return df_main

    def create_target(self, df, threshold=config.TARGET_THRESHOLD, look_ahead=config.LOOK_AHEAD_STEPS):
        if df.empty: return df
        df = df.copy()
        future_ret = df['close'].shift(-look_ahead) / df['close'] - 1
        conditions = [(future_ret < -threshold), (future_ret > threshold)]
        choices = [0, 2] 
        df['target_cls'] = np.select(conditions, choices, default=1)
        df['target_val'] = future_ret
        df.dropna(subset=['target_cls', 'target_val'], inplace=True)
        df['target_cls'] = df['target_cls'].astype(int)
        return df
    
    def get_precision_data(self, symbol=config.MAIN_SYMBOL):
        return self.fetch_data(symbol, config.TIMEFRAME_PRECISION, config.DATE_START, config.DATE_END)