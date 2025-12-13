import os
import multiprocessing
from datetime import datetime

# ---------------------------------------------------------
# [System] 하드웨어 및 시스템 자동 설정
# ---------------------------------------------------------
def get_optimal_cpu_cores():
    try:
        cores = multiprocessing.cpu_count()
        return max(2, cores - 1) 
    except:
        return 2

DETECTED_CORES = get_optimal_cpu_cores()

SYSTEM = {
    'MAIN_ML_DEVICE': 'cuda',      
    'MAIN_RL_DEVICE': 'cuda',      
    'OPT_TEACHER_DEVICE': 'cuda',
    'OPT_LOGIC_DEVICE': 'cpu',
    'NUM_WORKERS': DETECTED_CORES, 
    'SUPPRESS_WARNINGS': True       
}

# ---------------------------------------------------------
# [Path] 경로 설정 (수정됨)
# ---------------------------------------------------------
# 1. 현재 파일(config.py)이 있는 위치: .../tradingbot/core
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 프로젝트 루트 경로: .../tradingbot (한 단계 위로 이동)
BASE_DIR = os.path.dirname(CURRENT_DIR)

# 3. 하위 폴더 설정 (이제 루트 기준으로 생성됨)
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_BASE_DIR = os.path.join(BASE_DIR, 'logs')
MODEL_BASE_DIR = os.path.join(BASE_DIR, 'models_saved')

for d in [DATA_DIR, LOG_BASE_DIR, MODEL_BASE_DIR]:
    os.makedirs(d, exist_ok=True)

class SessionManager:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(LOG_BASE_DIR, self.session_id)
        self.model_dir = os.path.join(MODEL_BASE_DIR, self.session_id) # 학습된 모델은 여기 저장됨
        self.tensorboard_dir = os.path.join(self.log_dir, "tb_logs")

    def create(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        return {
            'id': self.session_id,
            'root': self.log_dir,
            'tb': self.tensorboard_dir,
            'model': self.model_dir,
            'log_file': os.path.join(self.log_dir, 'system.log')
        }

# ---------------------------------------------------------
# [Data] 데이터 대상 및 기간
# ---------------------------------------------------------
TARGET_COINS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
MAIN_SYMBOL = 'BTC/USDT'

TIMEFRAME_MAIN = '1h'
TIMEFRAME_AUX = '4h'   
TIMEFRAME_PRECISION = '5m'

DATE_START = '2019-01-01 00:00:00'
DATE_END   = '2025-12-31 00:00:00'
TEST_SPLIT_DATE = '2024-01-01 00:00:00'

EXCLUDE_COLS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 
    'target_cls', 'target_val', 'date', 'symbol'
]

# ---------------------------------------------------------
# [Feature] 지표 및 타겟
# ---------------------------------------------------------
INDICATOR_WINDOW = 12
LOOK_AHEAD_STEPS = 1           
TARGET_THRESHOLD = 0.005     

# ---------------------------------------------------------
# [Model] 모델 하이퍼파라미터
# ---------------------------------------------------------
XGB_PARAMS = {
    'n_estimators': 350,
    'max_depth': 8,
    'learning_rate': 0.0731,
    'n_jobs': -1,
    'random_state': 42,
    'eval_metric': 'mlogloss',
}

LSTM_PARAMS = {
    'units_1': 128,
    'units_2': 48,
    'dropout': 0.1377
}

ML_SEQ_LEN = 60
ML_EPOCHS = 150
ML_BATCH_SIZE = 1024

# [RL Reward Params] - Phase 3 최적화 결과 (2025-12-11 업데이트)
REWARD_PARAMS = {
    'profit_scale': 100,
    'teacher_bonus': 0.086,
    'teacher_penalty': 0.132,
    'mdd_penalty_factor': 0.503,
    'new_high_bonus': 0.380
}

# [RL PPO Params]
RL_TOTAL_TIMESTEPS = 10_000_000 
RL_PPO_PARAMS = {
    'learning_rate': 1.74e-5, # 최적화된 값
    'n_steps': 4096,
    'batch_size': 2048,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.005,
    'policy_kwargs': dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
}

# ---------------------------------------------------------
# [Trading] 거래 규칙 (Logic Optimization 대상)
# ---------------------------------------------------------
INITIAL_BALANCE = 10000.0
MAX_LEVERAGE = 20
FEE_RATE = 0.0006
SLIPPAGE = 0.0002

TRADING_RULES = {
    'trend_window': 200,       
    'sl_atr_multiplier': 2.0,  
    'risk_per_trade': 0.01,    
    'tp_trigger_atr': 2.2,
    'trailing_gap_atr': 3.0,   
    'min_trade_amount': 10.0,
    'funding_rate_hourly': 0.000025,
    'scale_down_factor': 0.5
}