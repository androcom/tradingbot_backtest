import sys
import os
import optuna
import numpy as np
import logging
from datetime import datetime

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import initialize_environment
initialize_environment()

from core import constants as config
from core import utils
from core.data_processor import DataProcessor
from execution.trade_engine import TradingCore
from learning.rl_environment import CryptoEnv
from core.config_manager import ConfigManager

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

logger = utils.get_logger("RLTuner")

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = os.path.join(config.LOG_BASE_DIR, 'optimization', f'RLOpt_{TIMESTAMP}.csv')
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

global_train_df = None

def load_data_once():
    global global_train_df
    if global_train_df is None:
        logger.info("⏳ Loading Data for RL Tuning...")
        dp = DataProcessor(logger)
        # 상세 로그 끄기
        dp.logger.setLevel(logging.ERROR)
        full_df = dp.get_ml_data(config.MAIN_SYMBOL)
        global_train_df = full_df[full_df.index < config.TEST_SPLIT_DATE].copy()
        logger.info(f"✅ Data Loaded: {len(global_train_df)} rows")
    return global_train_df

def objective(trial):
    df = load_data_once()
    reward_params = {
        'profit_scale': trial.suggest_int('rw_profit_scale', 100, 300, step=50),
        'teacher_bonus': trial.suggest_float('rw_teacher_bonus', 0.01, 0.1),
        'teacher_penalty': trial.suggest_float('rw_teacher_penalty', 0.05, 0.2),
        'mdd_penalty_factor': trial.suggest_float('rw_mdd_penalty', 0.5, 1.5),
        'new_high_bonus': trial.suggest_float('rw_new_high', 0.1, 1.0)
    }
    learning_rate = trial.suggest_float('ppo_lr', 1e-5, 3e-4, log=True)
    gamma = trial.suggest_categorical('ppo_gamma', [0.98, 0.99, 0.995])
    gae_lambda = trial.suggest_categorical('ppo_gae', [0.90, 0.95, 0.98])
    ent_coef = trial.suggest_float('ppo_ent', 0.001, 0.01, log=True)
    clip_range = trial.suggest_float('ppo_clip', 0.1, 0.3)
    
    def make_env():
        utils.silence_noisy_loggers()
        env = CryptoEnv(df, TradingCore('strategy_trend'), precision_df=None, debug=False)
        env.reward_params = reward_params 
        return Monitor(env)

    try:
        vec_env = SubprocVecEnv([make_env for _ in range(config.SYSTEM['NUM_WORKERS'])])
        env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=gamma)
        
        model = PPO(
            "MlpPolicy", env, learning_rate=learning_rate, n_steps=2048, batch_size=512,
            gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef, clip_range=clip_range,
            verbose=0, device=config.SYSTEM['MAIN_RL_DEVICE']
        )
        model.learn(total_timesteps=100_000)
        
        if len(model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep['r'] for ep in model.ep_info_buffer])
        else:
            mean_reward = -9999
    except:
        mean_reward = -9999
    finally:
        try: env.close()
        except: pass
        
    return mean_reward

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    logger.info("🚀 RL Param Optimization Started")
    load_data_once()
    
    DB_URL = f"sqlite:///{os.path.join(config.LOG_BASE_DIR, 'optuna_param.db')}"
    study = optuna.create_study(study_name=f"RL_{TIMESTAMP}", storage=DB_URL, direction='maximize', load_if_exists=True)
    
    try:
        study.optimize(objective, n_trials=30, show_progress_bar=True)
    finally:
        df_results = study.trials_dataframe()
        df_results.to_csv(CSV_PATH, index=False)
        logger.info(f"💾 Results saved to {CSV_PATH}")

    best = study.best_params
    logger.info(f"🏆 Best RL Params: {best}")
    
    cm = ConfigManager(config.MAIN_SYMBOL)
    
    best_reward_params = {
        'profit_scale': best['rw_profit_scale'],
        'teacher_bonus': best['rw_teacher_bonus'],
        'teacher_penalty': best['rw_teacher_penalty'],
        'mdd_penalty_factor': best['rw_mdd_penalty'],
        'new_high_bonus': best['rw_new_high']
    }
    best_ppo_params = {
        'learning_rate': best['ppo_lr'],
        'gamma': best['ppo_gamma'],
        'gae_lambda': best['ppo_gae'],
        'ent_coef': best['ppo_ent'],
        'clip_range': best['ppo_clip'],
        'n_steps': 2048, 'batch_size': 512, 'n_epochs': 10,
        'policy_kwargs': dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    }
    
    current_conf = cm.load_config()
    current_conf['rl_ppo_params'] = best_ppo_params
    current_conf['reward_params'] = best_reward_params
    cm.save_config(current_conf)
    logger.info("✅ RL PPO & Reward parameters updated in ConfigManager.")