import os
import sys
import time
import pandas as pd
import subprocess
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config, utils

utils.silence_noisy_loggers()
logger = utils.get_logger("ModelFactory")

class ModelFactory:
    def __init__(self, num_models=3):
        self.num_models = num_models
        self.results = []

    def run_factory(self):
        logger.info(f"🏭 Factory Started: Target {self.num_models} Models")
        
        for i in range(1, self.num_models + 1):
            name = f"Candidate_{i:02d}"
            logger.info(f"\n>>> [Phase {i}/{self.num_models}] {name}...")
            
            # Session Setup
            sm = config.SessionManager()
            paths = sm.create()
            
            # 1. Trainer
            logger.info(f"   Step 1: Training...")
            self._run_sub("pipelines/trainer.py", paths['id'])
            
            # 2. Logic
            logger.info(f"   Step 2: Logic Opt...")
            self._run_sub("optimizers/logic_tuner.py")
            
            # 3. Result
            self._collect(paths, name)

        logger.info("🎉 All Jobs Finished.")
        pd.DataFrame(self.results).to_csv(os.path.join(config.LOG_BASE_DIR, 'factory_report.csv'))

    def _run_sub(self, script, sid=None):
        env = os.environ.copy()
        if sid: env['FACTORY_SESSION_ID'] = sid
        subprocess.run([sys.executable, script], env=env, check=True)

    def _collect(self, paths, name):
        # Logic Result Parsing (Simplified)
        csv_path = os.path.join(config.LOG_BASE_DIR, 'logic_result.csv')
        bal = 0.0
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if not df.empty: bal = df['value'].max()
            
        self.results.append({'Model': name, 'ID': paths['id'], 'Balance': bal})
        logger.info(f"   ✅ {name} Done (Bal: ${bal:,.2f})")

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    NUM_MODELS = 3
    try:
        ModelFactory(NUM_MODELS).run_factory()
    except Exception as e:
        logger.error(f"Factory Fail: {e}")