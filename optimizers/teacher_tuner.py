import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import optuna
from optuna.integration import TFKerasPruningCallback # [NEW] 가지치기 콜백
import logging
import numpy as np
import pandas as pd
import gc
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config, utils
from core.data_loader import DataLoader

from xgboost import XGBClassifier
import tensorflow as tf 
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras import backend as K # type: ignore
from tensorflow.keras import mixed_precision # type: ignore

# Scikit-learn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

# -------------------------------------------------------------------------
# [가속화 1] Mixed Precision (RTX 3080 Ti 필수)
# -------------------------------------------------------------------------
mixed_precision.set_global_policy('mixed_float16')

utils.silence_noisy_loggers()
optuna.logging.set_verbosity(optuna.logging.ERROR)

logger = utils.get_logger("TeacherOpt")
DB_URL = utils.get_optuna_storage()
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = os.path.join(config.LOG_BASE_DIR, 'optimization', f'TeacherOpt_{TIMESTAMP}.csv')

RAW_DATA_MAIN = None
RAW_DATA_AUX = None

def load_raw_data():
    global RAW_DATA_MAIN, RAW_DATA_AUX
    if RAW_DATA_MAIN is not None: return
    logger.info("⏳ Loading Raw Data...")
    loader = DataLoader(logger)
    loader.logger.setLevel(logging.ERROR)
    RAW_DATA_MAIN = loader.fetch_data(config.MAIN_SYMBOL, config.TIMEFRAME_MAIN, config.DATE_START, config.DATE_END)
    RAW_DATA_AUX = loader.fetch_data(config.MAIN_SYMBOL, config.TIMEFRAME_AUX, config.DATE_START, config.DATE_END)
    loader.logger.setLevel(logging.INFO)
    logger.info(f"✅ Raw Data Loaded. (Main: {len(RAW_DATA_MAIN)} rows)")

def objective(trial):
    if RAW_DATA_MAIN is None: load_raw_data()
    
    window = trial.suggest_int('indicator_window', 12, 60, step=4)
    threshold = trial.suggest_float('target_threshold', 0.0015, 0.005, step=0.0005)
    
    loader = DataLoader(logger)
    loader.logger.setLevel(logging.CRITICAL) 
    
    df_main = loader._add_technical_indicators(RAW_DATA_MAIN, window, suffix='')
    df_aux = loader._add_technical_indicators(RAW_DATA_AUX, window, suffix='_4h')
    df_aux_res = df_aux.resample(config.TIMEFRAME_MAIN).ffill()
    aux_cols = [c for c in df_aux_res.columns if '_4h' in c]
    df_aux_feat = df_aux_res[aux_cols].shift(1)
    
    df = df_main.join(df_aux_feat).dropna()
    df = loader.create_target(df, threshold=threshold)
    
    if len(df) < 2000: return 0.0
    
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 400, step=50),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 8),
        'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.1),
        'n_jobs': 1,
        'random_state': 42,
        'device': config.SYSTEM['OPT_TEACHER_DEVICE'],
        'tree_method': 'hist',
        'verbosity': 0
    }
    
    lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 128, step=32)
    lstm_units_2 = trial.suggest_int('lstm_units_2', 16, 64, step=16)
    lstm_dropout = trial.suggest_float('lstm_dropout', 0.1, 0.4)
    
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    feat_cols = [c for c in df.columns if c not in config.EXCLUDE_COLS]
    X_raw = df[feat_cols].values
    y_raw = df['target_cls'].values.astype(int)
    scaler = RobustScaler()
    
    BATCH_SIZE = 1024 
    
    try:
        fold = 0
        for train_idx, test_idx in tscv.split(X_raw):
            fold += 1
            K.clear_session(); gc.collect()
            
            X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
            y_tr, y_te = y_raw[train_idx], y_raw[test_idx]
            
            scaler.fit(X_tr)
            X_tr_sc = scaler.transform(X_tr)
            X_te_sc = scaler.transform(X_te)
            
            classes = np.unique(y_tr)
            if len(classes) < 3: continue
            weights = compute_class_weight('balanced', classes=classes, y=y_tr)
            class_weights = dict(zip(classes, weights))
            sample_weights = np.array([class_weights[y] for y in y_tr])
            
            xgb = XGBClassifier(**xgb_params)
            xgb.fit(X_tr_sc, y_tr, sample_weight=sample_weights)
            p_xgb = xgb.predict_proba(X_te_sc)
            
            seq_len = config.ML_SEQ_LEN
            def mk_seq(d):
                if len(d) <= seq_len: return np.empty((0, seq_len, d.shape[1]))
                v = np.lib.stride_tricks.sliding_window_view(d, (seq_len, d.shape[1]))
                return v.squeeze(1) if v.ndim==4 else v
            
            X_tr_seq = mk_seq(X_tr_sc)
            X_te_seq = mk_seq(X_te_sc)
            y_tr_seq = y_tr[seq_len:]
            
            min_len_tr = min(len(X_tr_seq), len(y_tr_seq))
            X_tr_seq = X_tr_seq[:min_len_tr]
            y_tr_seq = y_tr_seq[:min_len_tr]
            
            if len(X_tr_seq) == 0: continue
            
            with tf.device('/GPU:0'):
                model = Sequential([
                    Input(shape=(seq_len, len(feat_cols))),
                    LSTM(lstm_units_1, return_sequences=True),
                    Dropout(lstm_dropout),
                    LSTM(lstm_units_2),
                    Dropout(lstm_dropout),
                    Dense(3, activation='softmax', dtype='float32') # [MP Safe]
                ])
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                # [가속화 2] Pruning Callback 추가
                # LSTM 학습 중 성능이 나쁘면 이 Trial 자체를 중단 (Epoch 단위 체크)
                # 주의: Fold가 여러 개이므로, 중간에 Pruning 되면 전체 Trial이 멈춤
                pruning_cb = TFKerasPruningCallback(trial, "accuracy")
                es = EarlyStopping(monitor='loss', patience=3)
                
                model.fit(
                    X_tr_seq, y_tr_seq, 
                    epochs=20, # Pruning 있으므로 Epoch 좀 줄여도 됨
                    batch_size=BATCH_SIZE, 
                    callbacks=[es, pruning_cb], # Pruning 추가
                    verbose=0, 
                    class_weight=class_weights
                )
                p_lstm = model.predict(X_te_seq, verbose=0, batch_size=BATCH_SIZE)
            
            min_len = min(len(p_xgb) - seq_len, len(p_lstm))
            if min_len <= 0: continue
            
            final_prob = (p_xgb[seq_len : seq_len + min_len] + p_lstm[:min_len]) / 2
            final_pred = np.argmax(final_prob, axis=1)
            y_true = y_te[seq_len : seq_len + min_len]
            
            score = precision_score(y_true, final_pred, average='macro', zero_division=0)
            scores.append(score)
            
            # [수정] Fold 단위 Pruning
            # KerasPruning은 Epoch 단위고, 여기는 Fold 단위 체크
            # 첫 번째 Fold 점수가 너무 낮으면 가망 없으므로 중단
            trial.report(score, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
        return np.mean(scores) if scores else 0.0
    except optuna.TrialPruned:
        raise # Optuna에게 Pruning 되었음을 알림
    except Exception as e:
        return 0.0

class TqdmCallback:
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="🚀 Teacher Opt", unit="trial", 
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Best: {postfix}]")
        self.best = 0.0
        # 폴더 생성 로직 삭제 (Utils가 처리함)

    def __call__(self, study, trial):
        if study.best_value > self.best: self.best = study.best_value
        self.pbar.set_postfix_str(f"{self.best:.4f}")
        self.pbar.update(1)
        
        # [표준화] Utils 함수 사용 (에러 처리, 폴더 생성 위임)
        utils.save_study_results(study, CSV_PATH)

if __name__ == "__main__":
    # [1] 표준 출력 버퍼링 해제 (로그 즉시 출력)
    sys.stdout.reconfigure(line_buffering=True)
    
    logger.info(f"🚀 Started on {config.SYSTEM['OPT_TEACHER_DEVICE']}")
    logger.info("⚡ Acceleration Enabled: Mixed Precision + Large Batch(1024)")
    
    # 데이터 로드 및 GPU 초기화
    load_raw_data()
    logger.info("⚙️ Initializing TensorFlow/GPU...")
    try:
        tf.config.list_physical_devices('GPU')
        with tf.device('/GPU:0'):
            tf.constant([1.0])
    except: pass
    
    dashboard_process = utils.launch_optuna_dashboard(logger)
    
    N_TRIALS = 500 
    study = optuna.create_study(
        study_name=f"Teacher_{TIMESTAMP}", 
        storage=DB_URL, 
        direction='maximize',
        load_if_exists=True
    )
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[TqdmCallback(N_TRIALS)])

    except KeyboardInterrupt:
        logger.info("\n🛑 Interrupted by User.")
        
    finally:
        # [핵심 수정] 결과 출력을 finally 블록으로 이동하여 무조건 실행되게 함
        logger.info(f"\n{'='*40}")
        if len(study.trials) > 0:
            logger.info(f"✅ Optimization Finished.")
            logger.info(f"💾 Result CSV: {CSV_PATH}")
            logger.info(f"🏆 Best Score: {study.best_value:.4f}")
            logger.info(f"🧩 Best Params: {study.best_params}")
        else:
            logger.warning("⚠️ No trials completed.")
        logger.info(f"{'='*40}")

        # 로그 강제 출력 (Flush)
        for handler in logger.handlers:
            handler.flush()
            
        # 대시보드 종료
        if dashboard_process:
            dashboard_process.terminate()
            logger.info("👋 Dashboard closed.")
            
        import time
        time.sleep(1) # 로그가 출력될 시간 확보