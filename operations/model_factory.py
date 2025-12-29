import sys
import os
import subprocess
import time
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import utils
from core import constants as config
from core.config_manager import ConfigManager
from operations.lifecycle_manager import ModelLifecycleManager

utils.silence_noisy_loggers()
logger = utils.get_logger("ModelFactory")

class ModelFactory:
    def __init__(self, symbol='BTC/USDT'):
        self.symbol = symbol
        self.loader = ConfigManager(symbol)
        
    def run_factory_process(self, target_strategy='strategy_trend'):
        start_time = time.time()
        logger.info(f"{'='*60}")
        logger.info(f"🏭 FACTORY START: {self.symbol} ({target_strategy})")
        logger.info(f"{'='*60}")

        from core.constants import SessionManager
        sm = SessionManager()
        paths = sm.create()
        session_id = paths['id']
        logger.info(f"🆔 Session ID: {session_id}")

        # [수정] utils의 get_worker_env 사용으로 통일
        env = utils.get_worker_env(session_id)
        
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Step 1: Model Tuning (Optional)
        # logger.info("\n>>> [Step 1] Running Model Tuner (Teacher)...")
        # tuner_script = os.path.join(BASE_DIR, 'tuning', 'model_tuner.py')
        # try:
        #     subprocess.run([sys.executable, tuner_script], env=env, check=True)
        # except subprocess.CalledProcessError:
        #     logger.error("❌ Model Tuning Failed. Continuing...")
            
        # Step 2: Training
        if target_strategy == 'strategy_trend':
            logger.info("\n>>> [Step 2] Running Train Worker...")
            trainer_script = os.path.join(BASE_DIR, 'learning', 'train_worker.py')
            try:
                subprocess.run([sys.executable, trainer_script], env=env, check=True)
            except subprocess.CalledProcessError:
                logger.error("❌ Training Failed.")
                return
        else:
            logger.info(f"\n>>> [Step 2] Skipping Training for {target_strategy}.")

        # Step 3: Strategy Tuning
        logger.info("\n>>> [Step 3] Running Strategy Tuner...")
        tuner_script = os.path.join(BASE_DIR, 'tuning', 'strategy_tuner.py')
        try:
            subprocess.run([sys.executable, tuner_script, target_strategy], env=env, check=True)
        except subprocess.CalledProcessError:
            logger.error("❌ Strategy Tuning Failed.")
            return

        # Step 4: Lifecycle
        logger.info("\n>>> [Step 4] Championship Battle...")
        lifecycle = ModelLifecycleManager(self.symbol)
        candidate_path = paths['model']
        lifecycle.battle(target_strategy, candidate_path)

        elapsed = time.time() - start_time
        logger.info(f"\n🎉 All Jobs Finished in {elapsed/60:.1f} min.")

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    factory = ModelFactory(config.MAIN_SYMBOL)
    factory.run_factory_process('strategy_trend')