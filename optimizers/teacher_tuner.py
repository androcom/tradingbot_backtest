import os
import sys

# [LOG BLOCKER]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import optuna
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from optuna.integration import TqdmCallback
from optuna.pruners import HyperbandPruner
from tensorflow.keras import mixed_precision # type: ignore
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config, utils
from core.data_loader import DataLoader
from models.hybrid_models import HybridLearner

# [가속화] Mixed Precision
mixed_precision.set_global_policy('mixed_float16')
utils.silence_noisy_loggers()

logger = utils.get_logger("TeacherOpt")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# 전역 데이터 캐싱
raw_data = None

def load_raw_data():
    global raw_data
    if raw_data is None:
        dl = DataLoader(logger)
        df = dl.get_ml_data(config.MAIN_SYMBOL)
        raw_data = df
        logger.info(f"✅ Data Loaded: {len(df)} rows")

def objective(trial):
    # 1. Hyperparameters Suggestion
    window = trial.suggest_int('indicator_window', 12, 60, step=4)
    threshold = trial.suggest_float('target_threshold', 0.0015, 0.005, step=0.0005)
    
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

    # 2. Data Preparation (On-the-fly)
    # 메모리 효율을 위해 전역 데이터 복사본 사용
    df = raw_data.copy()
    
    # 지표 재계산 로직 (실제 구현 시 config 변수 조작 대신, 직접 계산 함수 호출 필요)
    # 여기서는 간소화를 위해 가상 로직으로 대체하거나, config를 수정하지 않고
    # 모델 학습 파라미터 튜닝에 집중
    
    # Feature & Target 준비
    feature_cols = [c for c in df.columns if c not in config.EXCLUDE_COLS]
    
    # TimeSeriesSplit (3-Fold)
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        # Pruning Check (Fold 단위)
        trial.report(np.mean(scores) if scores else 0.0, fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Leakage Free Scaling
        scaler = RobustScaler()
        train_scaled = scaler.fit_transform(train_df[feature_cols])
        val_scaled = scaler.transform(val_df[feature_cols])
        
        # Sequence Generation
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
        
        # Train
        model = HybridLearner(None, logger) # 임시 모델 (저장 안함)
        model.xgb_params = xgb_params
        model.lstm_params = lstm_params
        model.batch_size = 1024 # [가속화]
        model.epochs = 20      # 튜닝용으로 에포크 축소
        
        # LSTM 내부 fit (verbose=0)
        model.build_lstm(input_shape=(config.ML_SEQ_LEN, len(feature_cols)))
        model.lstm_model.fit(X_seq_tr,  tf.keras.utils.to_categorical(y_tr, 3),
                             epochs=model.epochs, batch_size=model.batch_size, verbose=0)
        
        # XGBoost fit
        model.xgb_model.fit(X_flat_tr, y_tr, verbose=False)
        
        # Evaluate (mlogloss)
        preds = model.predict_proba(X_flat_val, X_seq_val)
        
        # Log Loss 계산 (직접 구현 or sklearn)
        from sklearn.metrics import log_loss
        try:
            score = log_loss(y_val, preds) # 낮을수록 좋음
            # Optuna는 maximize이므로 점수 반전 필요 -> Accuracy로 변경하거나 음수로 반환
            # 여기서는 Accuracy로 튜닝
            pred_cls = np.argmax(preds, axis=1)
            acc = np.mean(pred_cls == y_val)
            scores.append(acc)
        except:
            scores.append(0.0)
            
        # Clean up
        del model
        tf.keras.backend.clear_session()
        
    return np.mean(scores) if scores else 0.0

if __name__ == "__main__":
    # [LOG FIX] 출력 버퍼 즉시 비우기
    sys.stdout.reconfigure(line_buffering=True)
    
    logger.info(f"🚀 Started on {config.SYSTEM['OPT_TEACHER_DEVICE']}")
    logger.info("⚡ Acceleration Enabled: Mixed Precision + Batch(1024)")
    
    load_raw_data()
    
    # GPU Warm-up
    logger.info("⚙️ Initializing TensorFlow/GPU...")
    try:
        with tf.device('/GPU:0'):
            tf.constant([1.0])
    except: pass
    
    DB_PATH = os.path.join(config.LOG_BASE_DIR, 'optuna_study.db')
    DB_URL = f"sqlite:///{DB_PATH}"
    
    # Dashboard 실행 (No Browser, 0.0.0.0)
    dashboard_process = utils.launch_optuna_dashboard(logger, DB_URL)
    
    N_TRIALS = 500
    
    study = optuna.create_study(
        study_name=f"Teacher_{TIMESTAMP}", 
        storage=DB_URL, 
        direction='maximize',
        load_if_exists=True,
        pruner=HyperbandPruner() # [가속화] Pruning
    )
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[TqdmCallback(N_TRIALS)])
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Interrupted by User.")
        
    finally:
        # [LOG FIX] 종료 시 무조건 결과 출력
        logger.info(f"\n{'='*40}")
        if len(study.trials) > 0:
            logger.info(f"✅ Optimization Finished.")
            logger.info(f"🏆 Best Score: {study.best_value:.4f}")
            logger.info(f"🧩 Best Params: {study.best_params}")
        else:
            logger.warning("⚠️ No trials completed.")
        logger.info(f"{'='*40}")

        # [LOG FIX] 버퍼 강제 플러시
        for handler in logger.handlers:
            handler.flush()
        sys.stdout.flush()
            
        if dashboard_process:
            logger.info("👋 Closing Dashboard...")
            dashboard_process.terminate()