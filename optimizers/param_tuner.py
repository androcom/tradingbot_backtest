import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import logging
import optuna
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from core import utils
from core.data_loader import DataLoader
from core.trading_core import TradingCore
from core.rl_env import CryptoEnv

# 시스템 설정
utils.silence_noisy_loggers()
optuna.logging.set_verbosity(optuna.logging.ERROR)

logger = utils.get_logger("ParamOpt")
DB_URL = utils.get_optuna_storage()
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = os.path.join(config.LOG_BASE_DIR, 'optimization', f'ParamOpt_{TIMESTAMP}.csv')

# 데이터 로드 (전역 공유)
TRAIN_DF = None

def load_data_once():
    global TRAIN_DF
    if __name__ == "__main__":
        logger.info("⏳ Loading Data...")
    
    loader = DataLoader(logger)
    loader.logger.setLevel(logging.ERROR) # 상세 로그 숨기기
    full_df = loader.get_ml_data(config.MAIN_SYMBOL)
    train_df = full_df[full_df.index < config.TEST_SPLIT_DATE].copy()
    loader.logger.setLevel(logging.INFO)
    
    if __name__ == "__main__":
        logger.info(f"✅ Data Loaded: {len(train_df)} rows")
    return train_df

TRAIN_DF = load_data_once()

def objective(trial):
    # 1. 보상 파라미터
    reward_params = {
        'profit_scale': trial.suggest_int('profit_scale', 100, 300, step=50),
        'teacher_bonus': trial.suggest_float('teacher_bonus', 0.01, 0.1),
        'teacher_penalty': trial.suggest_float('teacher_penalty', 0.05, 0.2),
        'mdd_penalty_factor': trial.suggest_float('mdd_penalty', 0.5, 1.5),
        'new_high_bonus': trial.suggest_float('new_high_bonus', 0.1, 1.0)
    }
    
    # 2. RL 하이퍼파라미터
    lr = trial.suggest_float('learning_rate', 1e-5, 3e-4, log=True)
    
    def make_env():
        utils.silence_noisy_loggers() # Worker 내부 로그 차단
        env = CryptoEnv(TRAIN_DF, TradingCore(), precision_df=None, debug=False)
        env.reward_params = reward_params
        return Monitor(env)

    n_envs = config.SYSTEM['NUM_WORKERS']
    train_steps = 200_000 # 벤치마크 기반 최적 값
    
    mean_reward = -9999
    
    try:
        vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
        env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=0.99)
        
        model = PPO(
            "MlpPolicy", 
            env, 
            learning_rate=lr, 
            verbose=0, 
            device=config.SYSTEM['MAIN_RL_DEVICE'], 
            n_steps=4096, 
            batch_size=2048,
            ent_coef=0.01
        )
        
        model.learn(total_timesteps=train_steps)
        
        if len(model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep['r'] for ep in model.ep_info_buffer])
        else:
            mean_reward = -9999
            
    except Exception:
        mean_reward = -9999
    finally:
        try: env.close()
        except: pass

    return mean_reward

class TqdmCallback:
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="🚀 Param Opt", unit="trial", 
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Best: {postfix}]")
        self.best = -float('inf')
    def __call__(self, study, trial):
        if study.best_value > self.best: self.best = study.best_value
        self.pbar.set_postfix_str(f"{self.best:.2f}")
        self.pbar.update(1)
        utils.save_study_results(study, CSV_PATH)

if __name__ == "__main__":
    # [1] 로그 즉시 출력 설정
    sys.stdout.reconfigure(line_buffering=True)
    
    logger.info(f"🚀 Started on {config.SYSTEM['MAIN_RL_DEVICE']}")
    logger.info(f"💾 DB: {DB_URL}")
    
    # 50회 설정 (RL 학습이라 시간이 좀 걸림)
    N_TRIALS = 50 
    study = optuna.create_study(
        study_name=f"Param_{TIMESTAMP}", 
        storage=DB_URL, 
        direction='maximize',
        load_if_exists=True
    )
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[TqdmCallback(N_TRIALS)])
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Interrupted by User.")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        
    finally:
        # [핵심] 종료 시 무조건 실행되는 결과 리포트
        logger.info(f"\n{'='*40}")
        if len(study.trials) > 0:
            logger.info(f"✅ Optimization Finished.")
            logger.info(f"💾 Result CSV: {CSV_PATH}")
            logger.info(f"🏆 Best Reward: {study.best_value:.4f}")
            logger.info(f"🧩 Best Params: {study.best_params}")
        else:
            logger.warning("⚠️ No trials completed.")
        logger.info(f"{'='*40}")

        # 로그 강제 출력
        for handler in logger.handlers:
            handler.flush()
            
        import time
        time.sleep(1)