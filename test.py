import sys
import os
from core import utils
from core import constants as config
from operations.lifecycle_manager import ModelLifecycleManager

# ==========================================
# [설정] 멈췄던 세션 ID를 여기에 입력하세요
TARGET_SESSION_ID = "20251217_123520"  # <--- 폴더명 확인 후 수정 필수!
# ==========================================

def resume_step_4():
    # 환경 초기화
    utils.initialize_environment()
    logger = utils.get_logger("ResumeBattle")
    
    logger.info(f"🔄 Resuming Step 4 (Battle) for Session: {TARGET_SESSION_ID}")
    
    # 모델 경로 재구성
    candidate_path = os.path.join(config.MODEL_BASE_DIR, TARGET_SESSION_ID)
    
    if not os.path.exists(candidate_path):
        logger.error(f"❌ Path not found: {candidate_path}")
        return

    # Lifecycle Manager 가동
    lifecycle = ModelLifecycleManager(config.MAIN_SYMBOL)
    
    # 배틀 실행
    lifecycle.battle('strategy_trend', candidate_path)
    
    logger.info("✅ Resume Complete.")

if __name__ == "__main__":
    resume_step_4()