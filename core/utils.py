import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config

import logging
import subprocess
import warnings
import webbrowser
import pandas as pd

# 1. 환경 변수 설정 (C++ 레벨 로그 차단)
# 0 = all, 1 = INFO, 2 = INFO/WARN, 3 = INFO/WARN/ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # oneDNN 관련 알림 끄기
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' # 메모리 할당 로그 최소화

# 2. Python 경고 차단
warnings.filterwarnings("ignore")

def get_logger(name, log_file=None):
    """
    [표준 로거] 프로젝트 전체에서 동일한 로그 형식을 보장합니다.
    Format: [HH:MM:SS] [Name] Message
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False 

    if logger.handlers:
        logger.handlers = []

    formatter = logging.Formatter(f'[%(asctime)s] [{name}] %(message)s', datefmt='%H:%M:%S')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# [핵심] 이 함수를 호출하기 전에 환경변수가 설정되어야 가장 효과적입니다.
def silence_noisy_loggers():
    """
    TensorFlow, GPU, Abseil 등 시스템 로그를 강력하게 차단합니다.
    """
    # 1. 환경 변수 설정 (C++ 레벨 로그 차단)
    # 0 = all, 1 = INFO, 2 = INFO/WARN, 3 = INFO/WARN/ERROR
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # oneDNN 관련 알림 끄기
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' # 메모리 할당 로그 최소화
    
    # 2. Python 경고 차단
    warnings.filterwarnings("ignore")
    
    # 3. 라이브러리 로거 레벨 조정
    # absl은 TensorFlow 내부 로깅 라이브러리입니다.
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass

    noisy_loggers = [
        'werkzeug', 'tensorboard', 'tensorflow', 'absl',
        'h5py', 'matplotlib', 'urllib3', 'requests', 'optuna',
        'paramiko', 'nvgpu'
    ]
    
    for name in noisy_loggers:
        lg = logging.getLogger(name)
        lg.setLevel(logging.CRITICAL + 1) # CRITICAL보다 높은 레벨로 설정해 아예 안 뜨게 함
        lg.propagate = False

def get_optuna_storage():
    db_path = os.path.join(config.LOG_BASE_DIR, 'optuna_study.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return f"sqlite:///{db_path}"

def save_study_results(study, csv_path):
    """
    [표준 CSV 저장] Optuna 결과를 CSV로 저장합니다. (폴더 자동 생성 포함)
    """
    try:
        # 폴더가 없으면 생성
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        df = study.trials_dataframe()
        # 컬럼 순서 정리 (가독성)
        cols = df.columns.tolist()
        main_cols = ['number', 'value', 'state']
        param_cols = sorted([c for c in cols if c.startswith('params_')])
        other_cols = [c for c in cols if c not in main_cols and c not in param_cols]
        final_cols = main_cols + param_cols + other_cols
        final_cols = [c for c in final_cols if c in df.columns]
        
        df = df[final_cols]
        df.to_csv(csv_path, index=False)
    except Exception as e:
        # 에러 발생 시 출력 (tqdm 깨짐 방지 위해 print 사용 자제)
        logging.getLogger("Utils").warning(f"⚠️ Failed to save CSV: {e}")

def launch_optuna_dashboard(logger):
    """Optuna Dashboard 자동 실행"""
    db_url = get_optuna_storage()
    db_path = db_url.replace("sqlite:///", "")
    
    if not os.path.exists(db_path):
        logger.warning("⚠️ DB file not found yet. Dashboard might be empty.")

    try:
        process = subprocess.Popen(
            ["optuna-dashboard", db_url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        url = "http://127.0.0.1:8080"
        logger.info(f"📊 Optuna Dashboard started: {url}")
        try: webbrowser.open(url)
        except: pass
        return process
    except FileNotFoundError:
        logger.warning("⚠️ 'optuna-dashboard' not installed.")
        return None