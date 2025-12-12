import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config

import webbrowser
import tensorflow as tf
from tensorflow.keras import mixed_precision # type: ignore
from tensorboard import program

from core import utils
from core.config import SessionManager
from pipelines.trainer import PipelineTrainer

# 1. 노이즈 차단 (Utils 활용)
utils.silence_noisy_loggers()

# Mixed Precision 설정
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def launch_tensorboard(log_dir, logger):
    try:
        utils.silence_noisy_loggers() # TB 실행 전후로 차단
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir, '--port', '6006', '--reload_interval', '300'])
        url = tb.launch()
        utils.silence_noisy_loggers()
        logger.info(f">> TensorBoard started: {url}")
    except Exception as e:
        logger.warning(f">> TB Launch failed: {e}")

if __name__ == "__main__":
    # GPU 확인
    gpus = tf.config.list_physical_devices('GPU')
    gpu_msg = f"GPU: {gpus[0].name}" if gpus else "GPU: None"

    # 세션 생성
    session = SessionManager()
    paths = session.create()
    
    # [수정] 표준 로거 생성 (파일 저장 포함)
    logger = utils.get_logger("Main", log_file=paths['log_file'])
    
    logger.info(f"{'='*60}")
    logger.info(f" SESSION: {paths['id']} | {gpu_msg}")
    logger.info(f"{'='*60}")

    launch_tensorboard(config.LOG_BASE_DIR, logger)

    # 트레이너 실행
    trainer = PipelineTrainer(paths)
    trainer.run_all()