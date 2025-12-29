import sys
import os
import time
import schedule
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import utils
from core import constants as config
from operations.model_factory import ModelFactory

utils.silence_noisy_loggers()
logger = utils.get_logger("Scheduler")

def job_weekly_training():
    logger.info("⏰ Scheduled Job: Weekly Training Started")
    factory = ModelFactory(config.MAIN_SYMBOL)
    factory.run_factory_process('strategy_trend')

def job_daily_tuning():
    logger.info("⏰ Scheduled Job: Daily Logic Tuning Started")
    # Trend와 Range 전략의 로직만 가볍게 튜닝 (학습 제외)
    factory = ModelFactory(config.MAIN_SYMBOL)
    # run_factory_process를 수정하여 학습 없이 튜닝만 돌리는 모드 추가 필요하지만,
    # 현재는 Trend 전체 프로세스 실행
    factory.run_factory_process('strategy_trend')

if __name__ == "__main__":
    logger.info("⏳ Scheduler Started...")
    
    # 매주 월요일 새벽 2시에 모델 재학습
    schedule.every().monday.at("02:00").do(job_weekly_training)
    
    # 매일 아침 9시에 로직 튜닝 (선택 사항)
    # schedule.every().day.at("09:00").do(job_daily_tuning)
    
    while True:
        schedule.run_pending()
        time.sleep(60)