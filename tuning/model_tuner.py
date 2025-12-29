# [LOG SUPPRESSION] 최상단 배치
import os
import sys

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 환경 초기화 (로그 차단 등)
from core.utils import initialize_environment
initialize_environment()

import optuna
import numpy as np
import tensorflow as tf
from optuna.pruners import HyperbandPruner
from tensorflow.keras import mixed_precision # type: ignore
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from datetime import datetime

from core import constants as config
from core import utils
from core.data_processor import DataProcessor
from core.config_manager import ConfigManager
from learning.networks import HybridLearner

mixed_precision.set_global_policy('mixed_float16')
logger = utils.get_logger("ModelTuner")

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = os.path.join(config.LOG_BASE_DIR, 'optimization', f'TeacherOpt_{TIMESTAMP}.csv')
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

raw_data = None

def load_raw_data():
    global raw_data
    if raw_data is None:
        dp = DataProcessor(logger)
        df = dp.get_ml_data(config.MAIN_SYMBOL)
        raw_data = df
        logger.info(f"✅ Data Loaded: {len(df)} rows")
    return raw_data

def objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
        'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.1),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample', 0.6, 1.0),
        'n_jobs': 1 
    }
    lstm_params = {
        'units_1': trial.suggest_int('lstm_units_1', 32, 128, step=16),
        'units_2': trial.suggest_int('lstm_units_2', 16, 64, step=16),
        'dropout': trial.suggest_float('lstm_dropout', 0.1, 0.5)
    }

    df = load_raw_data().copy()
    feature_cols = [c for c in df.columns if c not in config.EXCLUDE_COLS]
    
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        trial.report(np.mean(scores) if scores else 0.0, fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        scaler = RobustScaler()
        train_scaled = scaler.fit_transform(train_df[feature_cols])
        val_scaled = scaler.transform(val_df[feature_cols])
        
        def create_seq(data, label):
            X_seq = np.lib.stride_tricks.sliding_window_view(data, window_shape=(config.ML_SEQ_LEN, data.shape[1]))
            if X_seq.ndim == 4: X_seq = X_seq.squeeze(axis=1)
            X_flat = data[config.ML_SEQ_LEN:]
            y = label[config.ML_SEQ_LEN:]
            min_len = min(len(X_seq), len(X_flat), len(y))
            return X_flat[:min_len], X_seq[:min_len], y[:min_len]

        X_flat_tr, X_seq_tr, y_tr = create_seq(train_scaled, train_df['target_cls'].values)
        X_flat_val, X_seq_val, y_val = create_seq(val_scaled, val_df['target_cls'].values)
        
        if len(y_tr) < 100 or len(y_val) < 100: continue
        
        model = HybridLearner(None, logger) 
        model.xgb_params = xgb_params
        model.lstm_params = lstm_params
        model.batch_size = 1024 
        model.epochs = 15
        
        # LSTM Build & Fit
        model.build_lstm(input_shape=(config.ML_SEQ_LEN, len(feature_cols)))
        # [수정] 정수형 타겟 사용 (sparse_categorical_crossentropy)
        model.lstm_model.fit(
            X_seq_tr, y_tr, 
            epochs=model.epochs, 
            batch_size=model.batch_size, 
            verbose=0
        )
        
        # XGB Build & Fit
        model.build_xgb()
        model.xgb_model.fit(X_flat_tr, y_tr, verbose=False)
        
        preds = model.predict_proba(X_flat_val, X_seq_val)
        pred_cls = np.argmax(preds, axis=1)
        acc = np.mean(pred_cls == y_val)
        scores.append(acc)
            
        del model
        tf.keras.backend.clear_session()
        
    return np.mean(scores) if scores else 0.0

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    logger.info(f"🚀 Teacher Optimization Started")
    load_raw_data()
    
    DB_URL = f"sqlite:///{os.path.join(config.LOG_BASE_DIR, 'optuna_teacher.db')}"
    
    study = optuna.create_study(
        study_name=f"Teacher_{TIMESTAMP}", 
        storage=DB_URL, 
        direction='maximize',
        load_if_exists=True,
        pruner=HyperbandPruner()
    )
    
    # Progress Bar 활성화 (Optuna 로그는 utils에서 껐음)
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    try:
        df_results = study.trials_dataframe()
        df_results.to_csv(CSV_PATH, index=False)
        logger.info(f"💾 Results saved to {CSV_PATH}")
    except: pass

    best = study.best_params
    logger.info(f"🏆 Best Params: {best}")
    
    cm = ConfigManager(config.MAIN_SYMBOL)
    xgb_best = {k.replace('xgb_', ''): v for k, v in best.items() if k.startswith('xgb_')}
    lstm_best = {k.replace('lstm_', ''): v for k, v in best.items() if k.startswith('lstm_')}
    xgb_best.update({'n_jobs': -1, 'random_state': 42, 'eval_metric': 'mlogloss'})
    
    cm.update_strategy_params('strategy_trend', {
        'xgb_params': xgb_best,
        'lstm_params': lstm_best
    })
    logger.info("✅ Strategy Trend parameters (AI Model) updated in ConfigManager.")