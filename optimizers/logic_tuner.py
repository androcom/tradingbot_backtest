import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import optuna
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import RobustScaler
import joblib 

# [통일] Mixed Precision 적용 (ML 추론 가속)
from tensorflow.keras import mixed_precision # type: ignore
mixed_precision.set_global_policy('mixed_float16')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from core import utils
from core.data_loader import DataLoader
from core.trading_core import TradingCore
from core.rl_env import CryptoEnv
from models.hybrid_models import HybridLearner

utils.silence_noisy_loggers()
optuna.logging.set_verbosity(optuna.logging.ERROR)

logger = utils.get_logger("LogicOpt")
DB_URL = utils.get_optuna_storage()
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = os.path.join(config.LOG_BASE_DIR, 'optimization', f'LogicOpt_{TIMESTAMP}.csv')

CACHED_DATA = {}

def prepare_simulation_data():
    logger.info("⏳ Loading Data & Models...")
    
    loader = DataLoader(logger)
    full_df = loader.get_ml_data(config.MAIN_SYMBOL)
    
    test_mask = full_df.index >= config.TEST_SPLIT_DATE
    test_df = full_df[test_mask].copy()
    
    # 모델 폴더 찾기
    if not os.path.exists(config.MODEL_BASE_DIR):
        raise FileNotFoundError(f"Model dir not found: {config.MODEL_BASE_DIR}")
    
    sessions = sorted([d for d in os.listdir(config.MODEL_BASE_DIR) 
                       if os.path.isdir(os.path.join(config.MODEL_BASE_DIR, d)) 
                       and not d.startswith('.')])
    if not sessions:
        raise FileNotFoundError("No trained models found.")
        
    latest_session = sessions[-1]
    model_dir = os.path.join(config.MODEL_BASE_DIR, latest_session)
    logger.info(f"   - Using Model: {latest_session}")
    
    # [수정] 저장된 Scaler 로드 (Leakage 방지)
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    feat_cols = [c for c in full_df.columns if c not in config.EXCLUDE_COLS]
    
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        data_scaled = scaler.transform(test_df[feat_cols])
    else:
        logger.warning("⚠️ Scaler not found, fitting new one (Leakage Risk)")
        scaler = RobustScaler()
        data_scaled = scaler.fit_transform(test_df[feat_cols])
    
    # ML Prediction
    ml_model = HybridLearner(model_dir)
    
    X_seq = np.lib.stride_tricks.sliding_window_view(data_scaled, window_shape=(config.ML_SEQ_LEN, len(feat_cols)))
    if X_seq.ndim == 4: X_seq = X_seq.squeeze(axis=1)
    X_flat = data_scaled[config.ML_SEQ_LEN:]
    
    min_len = min(len(X_seq), len(X_flat))
    X_seq = X_seq[:min_len]
    X_flat = X_flat[:min_len]
    
    logger.info("   - Generating ML Signals...")
    signals = ml_model.predict_proba(X_flat, X_seq)
    
    test_df_sim = test_df.iloc[config.ML_SEQ_LEN:].iloc[:min_len].copy()
    test_df_sim['ml_signal'] = signals
    
    # RL Prediction (Caching)
    logger.info("   - Caching RL Actions...")
    
    env = CryptoEnv(test_df_sim, TradingCore(), precision_df=None, debug=False)
    env = DummyVecEnv([lambda: env])
    
    vec_norm_path = os.path.join(model_dir, "vec_normalize.pkl")
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
        
    rl_agent = PPO.load(os.path.join(model_dir, "final_agent"))
    
    obs = env.reset()
    actions = []
    
    for _ in range(len(test_df_sim)):
        action, _ = rl_agent.predict(obs, deterministic=True)
        actions.append(action[0])
        obs, _, done, _ = env.step(action)
        if done: break
            
    CACHED_DATA['actions'] = np.array(actions)
    CACHED_DATA['timestamps'] = test_df_sim.index
    CACHED_DATA['closes'] = test_df_sim['close'].values
    CACHED_DATA['highs'] = test_df_sim['high'].values
    CACHED_DATA['lows'] = test_df_sim['low'].values
    CACHED_DATA['atrs'] = test_df_sim.get('atr', test_df_sim['close']*0.01).values
    CACHED_DATA['trends'] = test_df_sim.get('ema_trend_4h', test_df_sim['close']).values
    
    logger.info(f"✅ Simulation Data Ready: {len(actions)} steps")

def objective(trial):
    if not CACHED_DATA: prepare_simulation_data()
    
    sl_mult = trial.suggest_float('sl_atr_multiplier', 1.0, 5.0, step=0.1)
    risk_pct = trial.suggest_float('risk_per_trade', 0.01, 0.05, step=0.005)
    tp_trigger = trial.suggest_float('tp_trigger_atr', 1.0, 5.0, step=0.1)
    trailing_gap = trial.suggest_float('trailing_gap_atr', 0.5, 3.0, step=0.1)
    
    core = TradingCore()
    core.rules['sl_atr_multiplier'] = sl_mult
    core.rules['risk_per_trade'] = risk_pct
    core.rules['tp_trigger_atr'] = tp_trigger
    core.rules['trailing_gap_atr'] = trailing_gap
    
    actions = CACHED_DATA['actions']
    n = len(actions)
    
    for i in range(n):
        row = {
            'close': CACHED_DATA['closes'][i], 
            'high': CACHED_DATA['highs'][i], 
            'low': CACHED_DATA['lows'][i], 
            'ema_trend_4h': CACHED_DATA['trends'][i], 
            'atr': CACHED_DATA['atrs'][i]
        }
        core.process_step(actions[i], row, CACHED_DATA['timestamps'][i])
        
        if core.balance < 1000: break
            
    return core.balance

class TqdmCallback:
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="🚀 Logic Opt", unit="trial", 
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Best: ${postfix}]")
        self.best = -float('inf')
        
    def __call__(self, study, trial):
        if study.best_value > self.best: self.best = study.best_value
        self.pbar.set_postfix_str(f"{self.best:,.0f}")
        self.pbar.update(1)
        
        # [통일] Utils 함수 사용
        utils.save_study_results(study, CSV_PATH)

if __name__ == "__main__":
    # [1] 로그 즉시 출력 설정 (버퍼링 해제)
    sys.stdout.reconfigure(line_buffering=True)
    
    try:
        from optimizers.logic_tuner import prepare_simulation_data, objective 
        
        # 데이터 준비
        prepare_simulation_data()
        
        N_TRIALS = 5000
        study = optuna.create_study(
            study_name=f"Logic_{TIMESTAMP}", 
            storage=DB_URL, 
            direction='maximize',
            load_if_exists=True
        )
        
        logger.info(f"🚀 Started on {config.SYSTEM['OPT_LOGIC_DEVICE']}")
        
        # 최적화 실행
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[TqdmCallback(N_TRIALS)])
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Interrupted by User.")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    finally:
        # [핵심] 종료 시 무조건 실행되는 결과 리포트
        logger.info(f"\n{'='*40}")
        if 'study' in locals() and len(study.trials) > 0:
            logger.info(f"✅ Optimization Finished.")
            logger.info(f"💾 Result CSV: {CSV_PATH}")
            logger.info(f"🏆 Best Balance: ${study.best_value:,.2f}")
            logger.info(f"🧩 Best Params: {study.best_params}")
        else:
            logger.warning("⚠️ No trials completed.")
        logger.info(f"{'='*40}")

        # 로그 강제 출력
        for handler in logger.handlers:
            handler.flush()
            
        import time
        time.sleep(1)