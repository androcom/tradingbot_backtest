import os
import sys
import logging
import warnings
import optuna

def initialize_environment():
    """
    [초기화] 로그 차단 및 환경 변수 설정
    반드시 TensorFlow 등 무거운 라이브러리 import 전에 실행해야 함.
    """
    # 1. OS 환경 변수 설정 (TF 로드 전 필수)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['ABSL_LOG_LEVEL'] = 'error'

    # 2. 파이썬 경고 무시
    warnings.filterwarnings("ignore")

    # 3. Optuna 로그 정리
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 4. 기타 라이브러리 로그 차단
    silence_noisy_loggers()

def get_worker_env(session_id=None):
    """
    [Factory용] Subprocess에 전달할 깨끗한 환경 변수 딕셔너리 생성
    """
    env = os.environ.copy()
    
    # 필수 환경 변수 강제 주입
    env['TF_CPP_MIN_LOG_LEVEL'] = '3'
    env['TF_ENABLE_ONEDNN_OPTS'] = '0'
    env['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    env['ABSL_LOG_LEVEL'] = 'error'
    
    if session_id:
        env['FACTORY_SESSION_ID'] = session_id
        
    return env

def silence_noisy_loggers():
    noisy_loggers = [
        'werkzeug', 'tensorboard', 'tensorflow', 'absl',
        'h5py', 'matplotlib', 'urllib3', 'requests', 'optuna',
        'paramiko', 'nvgpu'
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.CRITICAL + 1)

def get_logger(name, log_file=None):
    """
    모든 모듈에서 공통으로 사용하는 로거 생성기
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)-10s] %(message)s', 
        datefmt='%H:%M:%S'
    )

    # 1. 콘솔 핸들러
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 2. 파일 핸들러
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger