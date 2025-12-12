import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import shutil
import logging
import pandas as pd
import optuna
from datetime import datetime

from core import utils
from pipelines.trainer import PipelineTrainer
from optimizers.param_tuner import objective as param_objective
from optimizers.logic_tuner import objective as logic_objective, prepare_simulation_data

# 로깅 설정
sys.stdout.reconfigure(line_buffering=True)
utils.silence_noisy_loggers()
log_path = os.path.join(config.LOG_BASE_DIR, "factory_system.log")
logger = utils.get_logger("Factory", log_file=log_path)

class ModelFactory:
    def __init__(self, num_models=5):
        self.num_models = num_models
        self.report_path = os.path.join(config.LOG_BASE_DIR, "factory_report.csv")
        self.results = []

    def run_factory(self):
        logger.info(f"🏭 Starting Model Factory: Producing {self.num_models} Models...")

        for i in range(1, self.num_models + 1):
            model_name = f"Candidate_{i:02d}_{datetime.now().strftime('%H%M')}"
            logger.info(f"\n{'='*60}")
            logger.info(f"🔨 Processing Model {i}/{self.num_models}: {model_name}")
            logger.info(f"{'='*60}")

            # -----------------------------------------------------
            # Step 1: 파라미터 튜닝 (Hyperparameter Tuning)
            # -----------------------------------------------------
            logger.info("   [Step 1] Optimizing Hyperparameters...")
            study_param = optuna.create_study(direction='maximize')
            # 시간 단축을 위해 Trial 횟수 조절
            study_param.optimize(param_objective, n_trials=25) 
            
            best_reward_params = study_param.best_params
            logger.info(f"   ✅ Best Params Found: {best_reward_params}")

            # Config 메모리 업데이트 (파일은 건드리지 않음)
            self._apply_params_to_config(best_reward_params)

            # -----------------------------------------------------
            # Step 2: 메인 모델 학습 (Main Training)
            # -----------------------------------------------------
            logger.info("   [Step 2] Training Main Model (This takes long)...")
            
            # 세션 생성
            session = config.SessionManager()
            paths = session.create()
            
            # 학습 실행
            trainer = PipelineTrainer(paths)
            trainer.run_all()
            
            # -----------------------------------------------------
            # Step 3: 로직 최적화 (Logic Tuning)
            # -----------------------------------------------------
            logger.info("   [Step 3] Optimizing Trading Logic...")
            
            # 방금 학습한 모델 경로로 config 임시 수정하여 로직 튜너가 인식하게 함
            # (Logic Tuner는 가장 최근 모델을 가져오므로 별도 조치 불필요하나 안전장치)
            prepare_simulation_data() # 캐시 갱신
            
            study_logic = optuna.create_study(direction='maximize')
            study_logic.optimize(logic_objective, n_trials=2500)
            
            best_logic = study_logic.best_params
            best_balance = study_logic.best_value
            logger.info(f"   ✅ Best Logic: Balance ${best_balance:,.2f}")

            # -----------------------------------------------------
            # Step 4: 모델 저장 및 기록
            # -----------------------------------------------------
            save_dir = os.path.join(config.MODEL_BASE_DIR, model_name)
            
            # 학습된 모델 이동
            if os.path.exists(save_dir): shutil.rmtree(save_dir)
            shutil.copytree(paths['model'], save_dir)
            
            # 결과 기록
            result = {
                "Model": model_name,
                "Final_Balance": best_balance,
                "Reward_Params": str(best_reward_params),
                "Logic_Params": str(best_logic),
                "Path": save_dir
            }
            self.results.append(result)
            self._save_report()
            
            logger.info(f"🎉 Model {model_name} Completed.")

    def _apply_params_to_config(self, params):
        """Optuna에서 찾은 파라미터를 현재 메모리의 config에 적용"""
        # REWARD_PARAMS 업데이트
        if not hasattr(config, 'REWARD_PARAMS'):
            config.REWARD_PARAMS = {}
        
        # PPO Learning Rate 등 별도 키가 있다면 분기 처리
        for key, value in params.items():
            if key == 'learning_rate':
                config.RL_PPO_PARAMS['learning_rate'] = value
            elif key in ['profit_scale', 'teacher_bonus', 'teacher_penalty', 'mdd_penalty_factor', 'new_high_bonus']:
                config.REWARD_PARAMS[key] = value

    def _save_report(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.report_path, index=False)
        logger.info(f"📄 Report updated: {self.report_path}")

if __name__ == "__main__":
    # 몇 개의 샘플 모델을 만들지 설정 (예: 3개)
    # 1개당 약 24~30시간 소요되므로, 주말 내내 돌리려면 2~3개 추천
    factory = ModelFactory(num_models=3)
    factory.run_factory()