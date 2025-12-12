import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config

import shutil
import logging
import time
from datetime import datetime
import schedule # pip install schedule 필요


from core import utils
from pipelines.trainer import PipelineTrainer
from pipelines.evaluator import ModelEvaluator
from optimizers.logic_tuner import objective as logic_objective # 기존 로직 튜너 활용
import optuna

# [수정] 로거 설정
log_path = os.path.join(config.LOG_BASE_DIR, "auto_manager.log")
logger = utils.get_logger("AutoManager", log_file=log_path)

# [수정] 노이즈 차단
utils.silence_noisy_loggers()

class AutoManager:
    def __init__(self):
        self.champ_dir = os.path.join(config.MODEL_BASE_DIR, "champion")
        self.chall_dir = os.path.join(config.MODEL_BASE_DIR, "challenger")
        self.evaluator = ModelEvaluator()

    def run_daily_logic_tuning(self):
        """[매일] 매매 로직(SL/TP) 최적화"""
        logger.info(">>> Starting Daily Logic Optimization...")
        
        # 1. Optuna 실행 (Logic Tuner)
        study = optuna.create_study(direction='maximize')
        study.optimize(logic_objective, n_trials=100) # 매일 100번만 가볍게
        
        best_params = study.best_params
        logger.info(f"✅ Daily Optimization Done. Best Params: {best_params}")
        
        # 2. Config 업데이트 (실제로는 JSON 파일 등에 저장하여 로드하는 방식 권장)
        # 여기서는 로그만 남김
        self._update_config_file(best_params)

    def run_weekly_model_training(self):
        """[매주] 새로운 모델 학습 (Challenger 생성)"""
        logger.info(">>> Starting Weekly Model Training (Challenger)...")
        
        # 1. 세션 생성 및 학습
        session = config.SessionManager()
        paths = session.create() # 임시 폴더 생성
        
        # Challenger 학습
        trainer = PipelineTrainer(paths)
        trainer.run_all()
        
        # 2. 학습된 모델을 Challenger 폴더로 이동
        if os.path.exists(self.chall_dir):
            shutil.rmtree(self.chall_dir)
        shutil.copytree(paths['model'], self.chall_dir)
        
        logger.info("✅ Challenger Model Trained & Saved.")
        
        # 3. 승부 (Champion vs Challenger)
        self._run_battle()

    def _run_battle(self):
        if not os.path.exists(self.champ_dir):
            # 챔피언이 없으면 도전자가 바로 챔피언 등극
            logger.info("No Champion found. Challenger becomes the first Champion.")
            shutil.copytree(self.chall_dir, self.champ_dir)
            return

        winner = self.evaluator.battle(self.champ_dir, self.chall_dir)
        
        if winner == "challenger":
            # 챔피언 교체 (백업 후 덮어쓰기)
            backup_name = f"champion_backup_{datetime.now().strftime('%Y%m%d')}"
            shutil.move(self.champ_dir, os.path.join(config.MODEL_BASE_DIR, backup_name))
            shutil.move(self.chall_dir, self.champ_dir)
            logger.info(f"👑 Model Swapped! Old champion backed up to {backup_name}")
        else:
            # 도전자 폐기
            shutil.rmtree(self.chall_dir)
            logger.info("🗑️ Challenger Discarded.")

    def _update_config_file(self, params):
        # 실제 구현 시: config.json을 쓰고 config.py가 그걸 읽게 수정 필요
        logger.info(f"Update Config Request: {params}")

# ---------------------------------------------------------
# 실행 루프
# ---------------------------------------------------------
if __name__ == "__main__":
    manager = AutoManager()
    
    # 스케줄 설정
    # 매일 아침 9시 로직 튜닝
    schedule.every().day.at("09:00").do(manager.run_daily_logic_tuning)
    
    # 매주 월요일 새벽 2시 모델 재학습
    schedule.every().monday.at("02:00").do(manager.run_weekly_model_training)
    
    logger.info("🚀 Auto-Manager Started. Waiting for schedule...")
    
    # 테스트용: 시작하자마자 한 번 실행해보고 싶다면 주석 해제
    # manager.run_daily_logic_tuning()
    
    while True:
        schedule.run_pending()
        time.sleep(60)