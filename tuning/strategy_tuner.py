import os
import sys
import optuna
import numpy as np
import joblib
from datetime import datetime

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import initialize_environment
initialize_environment()

from core import constants as config
from core import utils
from core.data_processor import DataProcessor
from core.config_manager import ConfigManager
from strategies.trend_strategy import TrendStrategy
from strategies.range_strategy import RangeStrategy
from strategies.defense_strategy import DefenseStrategy
from learning.networks import HybridLearner

logger = utils.get_logger("StrategyTuner")

cached_data = {} 
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = os.path.join(config.LOG_BASE_DIR, 'optimization', f'StrategyOpt_{TIMESTAMP}.csv')
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

def load_data_and_prepare(strategy_name):
    global cached_data
    if strategy_name in cached_data: return cached_data[strategy_name]
    
    logger.info(f"⏳ Preparing Data for {strategy_name}...")
    dp = DataProcessor(logger)
    df = dp.get_ml_data(config.MAIN_SYMBOL)
    test_mask = df.index >= config.TEST_SPLIT_DATE
    val_df = df[test_mask].copy()
    
    if strategy_name == 'strategy_trend':
        model_path = _get_latest_model_path()
        if model_path:
            logger.info(f"   - Loading AI Model from {os.path.basename(model_path)}")
            val_df = _inject_ai_signals(val_df, model_path, df)
        else:
            logger.warning("⚠️ No AI Model found. Signals will be 0.")
    elif strategy_name == 'strategy_range':
        if 'rsi' not in val_df.columns:
            import ta
            val_df['rsi'] = ta.momentum.RSIIndicator(val_df['close']).rsi()

    cached_data[strategy_name] = val_df
    return val_df

def _get_latest_model_path():
    base_dir = config.MODEL_BASE_DIR
    if not os.path.exists(base_dir): return None
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.replace('_', '').isdigit()]
    if not subdirs: return None
    return max(subdirs, key=os.path.getmtime)

def _inject_ai_signals(val_df, model_path, full_df):
    scaler_path = os.path.join(model_path, "scaler.pkl")
    if not os.path.exists(scaler_path): return val_df
    
    scaler = joblib.load(scaler_path)
    feature_cols = [c for c in full_df.columns if c not in config.EXCLUDE_COLS]
    data_scaled = scaler.transform(val_df[feature_cols])
    
    X_seq = np.lib.stride_tricks.sliding_window_view(data_scaled, window_shape=(config.ML_SEQ_LEN, len(feature_cols)))
    if X_seq.ndim == 4: X_seq = X_seq.squeeze(axis=1)
    X_flat = data_scaled[config.ML_SEQ_LEN:]
    
    min_len = min(len(X_seq), len(X_flat))
    model = HybridLearner(model_path, logger)
    if model.load():
        preds = model.predict_proba(X_flat[:min_len], X_seq[:min_len])
        if preds.ndim > 1: sigs = preds[:, 1] - preds[:, 0]
        else: sigs = preds
        val_df['ml_signal'] = 0.0
        val_df.iloc[-len(sigs):, val_df.columns.get_loc('ml_signal')] = sigs
    return val_df

def run_simulation(strategy_cls, df, params, strategy_name):
    # Mock ConfigManager for simulation
    class MockCM:
        def load_config(self):
            return { strategy_name: {'trading_rules': params} }
    
    mock_cm = MockCM()
    strategy = strategy_cls(mock_cm)
    balance = config.INITIAL_BALANCE
    ml_signals = df['ml_signal'].values if 'ml_signal' in df.columns else np.zeros(len(df))
    
    # Simulation Loop
    for i in range(len(df)):
        row = df.iloc[i]
        action = 0
        if strategy_name == 'strategy_trend':
            sig = ml_signals[i]
            th = params.get('threshold_entry', 0.5)
            if sig > th: action = 1
            elif sig < -th: action = 2
        
        balance, _ = strategy.process(row, action, balance)
        if balance < config.INITIAL_BALANCE * 0.5: break
            
    return balance

def objective(trial, strategy_name):
    df = load_data_and_prepare(strategy_name)
    params = {}
    strategy_cls = None
    
    if strategy_name == 'strategy_trend':
        strategy_cls = TrendStrategy
        params['threshold_entry'] = trial.suggest_float('threshold_entry', 0.2, 0.8, step=0.05)
        params['tp_rate'] = trial.suggest_float('tp_rate', 0.03, 0.15, step=0.005)
        params['sl_rate'] = trial.suggest_float('sl_rate', 0.01, 0.05, step=0.002)
        params['trailing_gap_rate'] = params['tp_rate'] * 0.5 
    elif strategy_name == 'strategy_range':
        strategy_cls = RangeStrategy
        params['rsi_lower'] = trial.suggest_int('rsi_lower', 20, 40)
        params['rsi_upper'] = trial.suggest_int('rsi_upper', 60, 80)
        params['tp_rate'] = trial.suggest_float('tp_rate', 0.005, 0.04)
        params['sl_rate'] = trial.suggest_float('sl_rate', 0.01, 0.05)
    elif strategy_name == 'strategy_defense':
        strategy_cls = DefenseStrategy
        params['defense_factor'] = trial.suggest_float('defense_factor', 0.8, 1.0)

    final_balance = run_simulation(strategy_cls, df, params, strategy_name)
    if final_balance < config.INITIAL_BALANCE:
        return final_balance - (config.INITIAL_BALANCE - final_balance)
    return final_balance

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    target_strategy = sys.argv[1] if len(sys.argv) > 1 else 'strategy_trend'
    logger.info(f"🚀 Logic Optimization Target: {target_strategy}")
    
    DB_URL = f"sqlite:///{os.path.join(config.LOG_BASE_DIR, 'optuna_logic.db')}"
    study_name = f"{target_strategy}_{datetime.now().strftime('%Y%m')}"
    
    study = optuna.create_study(study_name=study_name, storage=DB_URL, direction='maximize', load_if_exists=True)
    
    try:
        study.optimize(lambda t: objective(t, target_strategy), n_trials=50, show_progress_bar=True)
    finally:
        df_results = study.trials_dataframe()
        df_results.to_csv(CSV_PATH, index=False)
        logger.info(f"💾 Results saved to {CSV_PATH}")
    
    best = study.best_params
    logger.info(f"🏆 Best Params: {best}")
    
    cm = ConfigManager(config.MAIN_SYMBOL)
    cm.update_strategy_params(target_strategy, {'trading_rules': best})
    logger.info(f"✅ Config Updated for {target_strategy}.")