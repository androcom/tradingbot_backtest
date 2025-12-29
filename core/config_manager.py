import json
import os
import logging
from . import constants as config

class ConfigManager:
    def __init__(self, symbol='BTC/USDT'):
        self.symbol = symbol
        self.safe_symbol = symbol.replace('/', '_')
        
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_dir = os.path.join(self.root_dir, 'configs')
        self.config_path = os.path.join(self.config_dir, f"{self.safe_symbol}.json")
        
        self.logger = logging.getLogger(f"ConfigManager_{self.safe_symbol}")
        os.makedirs(self.config_dir, exist_ok=True)

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load config: {e}. Using defaults.")
                return self._create_default_config()
        else:
            self.logger.info("Config file not found. Creating default.")
            return self._create_default_config()

    def _create_default_config(self):
        default_conf = {
            'symbol': self.symbol,
            'last_updated': "init",
            
            # [Strategy A] Trend Following (XGB+LSTM)
            'strategy_trend': {
                'active': True,
                'xgb_params': config.XGB_PARAMS,
                'lstm_params': config.LSTM_PARAMS,
                'trading_rules': config.TRADING_RULES,
                'indicators': {
                    'window': config.INDICATOR_WINDOW,
                    'threshold': config.TARGET_THRESHOLD
                }
            },
            
            # [Strategy B] Mean Reversion (RSI/Bollinger)
            'strategy_range': {
                'active': False,
                'rsi_period': 14,
                'rsi_lower': 30,
                'rsi_upper': 70,
                'bb_window': 20,
                'tp_rate': 0.01,
                'sl_rate': 0.01
            },
            
            # [Strategy C] Defense
            'strategy_defense': {
                'active': False,
                'defense_factor': 0.95
            },
            
            # [RL Manager]
            'rl_manager': {
                'model_path': None,
                'lookback_window': 24
            }
        }
        self.save_config(default_conf)
        return default_conf

    def save_config(self, new_config):
        try:
            with open(self.config_path, 'w') as f:
                json.dump(new_config, f, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")

    def update_strategy_params(self, strategy_name, new_params):
        conf = self.load_config()
        if strategy_name not in conf:
            conf[strategy_name] = {}
        
        for k, v in new_params.items():
            conf[strategy_name][k] = v
            
        self.save_config(conf)