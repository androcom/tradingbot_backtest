import os
import sys
import logging
import subprocess
import warnings

def silence_noisy_loggers():
    """
    TensorFlow, GPU, Abseil, Matplotlib 등 시스템 로그 차단
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    warnings.filterwarnings("ignore")
    
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
        lg.setLevel(logging.CRITICAL + 1)
        lg.propagate = False

def get_logger(name, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] %(message)s', 
        datefmt='%H:%M:%S'
    )

    # Console Handler (즉시 출력 설정)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def launch_optuna_dashboard(logger, storage_url):
    """
    Optuna Dashboard를 백그라운드에서 실행 (0.0.0.0 바인딩)
    """
    try:
        # 이미 실행 중인지 확인 (포트 충돌 방지 로직은 생략, 덮어쓰기)
        cmd = [
            "optuna-dashboard", 
            storage_url, 
            "--host", "0.0.0.0",  # 외부 접속 허용
            "--port", "8080",     # 고정 포트
            "--no-browser"        # 브라우저 자동 실행 방지
        ]
        # 로그를 devnull로 보내서 터미널 오염 방지
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        logger.info(f"📊 Optuna Dashboard: http://localhost:8080 (Remote Accessible)")
        return process
    except Exception as e:
        logger.warning(f"Failed to launch dashboard: {e}")
        return None