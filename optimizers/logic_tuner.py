import os
import sys
import optuna
import pandas as pd
import numpy as np
import joblib

# [LOG BLOCKER]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config, utils
from core.data_loader import DataLoader
from models.hybrid_models import HybridLearner
from sklearn.preprocessing import RobustScaler

utils.silence_noisy_loggers()
logger = utils.get_logger("LogicOpt")
cached_sim_data = None

def load_latest_model():
    base = config.MODEL_BASE_DIR
    dirs = [os.path.join(base, d) for d in os.listdir(base) if d.replace('_','').isdigit()]
    return max(dirs, key=os.path.getmtime) if dirs else None

def prepare_simulation_data():
    global cached_sim_data
    logger.info("⏳ Preparing Data...")
    df = DataLoader(logger).get_ml_data(config.MAIN_SYMBOL)
    test_df = df[df.index >= config.TEST_SPLIT_DATE].copy()
    
    model_path = load_latest_model()
    if not model_path: raise FileNotFoundError("No models")
    
    ml = HybridLearner(model_path, logger)
    if not ml.load(): raise Exception("Load failed")
    
    scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
    feat = [c for c in test_df.columns if c not in config.EXCLUDE_COLS]
    
    data = scaler.transform(test_df[feat])
    X_seq = np.lib.stride_tricks.sliding_window_view(data, window_shape=(config.ML_SEQ_LEN, len(feat)))
    if X_seq.ndim == 4: X_seq = X_seq.squeeze(1)
    X_flat = data[config.ML_SEQ_LEN:]
    
    # [Fix] Net Signal
    preds = ml.predict_proba(X_flat, X_seq)
    sig = preds[:, 1] - preds[:, 0] if preds.ndim > 1 else preds
    
    test_df['ml_signal'] = 0.0
    test_df.iloc[config.ML_SEQ_LEN:, test_df.columns.get_loc('ml_signal')] = sig
    cached_sim_data = test_df
    logger.info("✅ Data Ready")

def objective(trial):
    if cached_sim_data is None: prepare_simulation_data()
    
    # Params
    th_ent = trial.suggest_float('threshold_entry', 0.1, 0.8, step=0.05)
    tp = trial.suggest_float('tp_rate', 0.005, 0.05)
    sl = trial.suggest_float('sl_rate', 0.005, 0.03)
    
    # Sim
    bal = config.INITIAL_BALANCE
    pos = 0; entry = 0.0
    
    closes = cached_sim_data['close'].values
    highs = cached_sim_data['high'].values
    lows = cached_sim_data['low'].values
    sigs = cached_sim_data['ml_signal'].values
    
    for i in range(len(closes)):
        p = closes[i]
        if pos == 1:
            if highs[i] >= entry*(1+tp): bal*=(1+tp); pos=0
            elif lows[i] <= entry*(1-sl): bal*=(1-sl); pos=0
        elif pos == -1:
            if lows[i] <= entry*(1-tp): bal*=(1+tp); pos=0
            elif highs[i] >= entry*(1+sl): bal*=(1-sl); pos=0
        
        if pos == 0:
            if sigs[i] > th_ent: pos=1; entry=p
            elif sigs[i] < -th_ent: pos=-1; entry=p
            
    return bal if bal > config.INITIAL_BALANCE else bal - (config.INITIAL_BALANCE - bal)

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    DB_URL = f"sqlite:///{os.path.join(config.LOG_BASE_DIR, 'optuna_logic.db')}"
    
    try:
        prepare_simulation_data()
        study = optuna.create_study(study_name=f"Logic_{config.TIMESTAMP}", storage=DB_URL, direction='maximize', load_if_exists=True)
        study.optimize(objective, n_trials=1000, n_jobs=4)
    except KeyboardInterrupt: pass
    except Exception as e: logger.error(e)
    finally:
        logger.info(f"\n{'='*40}")
        if 'study' in locals() and len(study.trials) > 0:
            logger.info(f"🏆 Best: ${study.best_value:,.2f}")
            pd.DataFrame(study.trials_dataframe()).to_csv(os.path.join(config.LOG_BASE_DIR, 'logic_result.csv'))
        logger.info(f"{'='*40}")
        for h in logger.handlers: h.flush()