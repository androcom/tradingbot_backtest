import os
import shutil
import json
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import utils
from core import constants as config
from core.config_manager import ConfigManager
from learning.evaluator import ModelEvaluator

utils.silence_noisy_loggers()
logger = utils.get_logger("LifecycleMgr")

class ModelLifecycleManager:
    def __init__(self, symbol='BTC/USDT'):
        self.symbol = symbol
        self.base_dir = config.MODEL_BASE_DIR
        self.registry_path = os.path.join(self.base_dir, 'registry.json')
        self.loader = ConfigManager(symbol)
        self.registry = self._load_registry()

    def _load_registry(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_registry(self):
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=4)

    def register_candidate(self, strategy_name, model_path, score):
        if strategy_name not in self.registry:
            self.registry[strategy_name] = {'champion': None, 'history': []}
            
        model_id = os.path.basename(model_path)
        record = {
            'id': model_id,
            'path': model_path,
            'score': float(score),
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.registry[strategy_name]['history'].append(record)
        self._save_registry()
        logger.info(f"📝 Registered Candidate for {strategy_name}: {model_id} (Score: {score:.2f})")
        return record

    def promote_champion(self, strategy_name, candidate_record):
        strategy_dir = os.path.join(self.base_dir, strategy_name)
        champion_dir = os.path.join(strategy_dir, 'champion')
        
        if os.path.exists(champion_dir):
            archive_dir = os.path.join(strategy_dir, 'archive', f"old_{datetime.now().strftime('%Y%m%d%H%M')}")
            os.makedirs(os.path.dirname(archive_dir), exist_ok=True)
            shutil.move(champion_dir, archive_dir)
            self._cleanup_archive(os.path.join(strategy_dir, 'archive'))

        shutil.copytree(candidate_record['path'], champion_dir)
        self.registry[strategy_name]['champion'] = candidate_record
        self._save_registry()
        
        self.loader.update_strategy_params(strategy_name, {
            'model_path': champion_dir,
            'model_score': candidate_record['score'],
            'last_updated': candidate_record['date']
        })
        logger.info(f"👑 NEW CHAMPION PROMOTED for {strategy_name}!")

    def battle(self, strategy_name, candidate_path):
        logger.info(f"⚔️ BATTLE START: {strategy_name}")
        evaluator = ModelEvaluator()
        conf = self.loader.load_config()
        
        challenger_roi, _ = evaluator.evaluate_model(candidate_path, config_override=conf)
        champ_record = self.registry.get(strategy_name, {}).get('champion')
        
        if not champ_record:
            logger.info("   Thinking: No Champion exists. Challenger wins by default.")
            record = self.register_candidate(strategy_name, candidate_path, challenger_roi)
            self.promote_champion(strategy_name, record)
            return True

        champ_path = os.path.join(self.base_dir, strategy_name, 'champion')
        champ_roi, _ = evaluator.evaluate_model(champ_path, config_override=conf)
        
        logger.info(f"   🏆 Champion ROI: {champ_roi:.2f}%")
        logger.info(f"   🥊 Challenger ROI: {challenger_roi:.2f}%")
        
        if challenger_roi > champ_roi * 1.05:
            logger.info("🎉 Challenger Wins!")
            record = self.register_candidate(strategy_name, candidate_path, challenger_roi)
            self.promote_champion(strategy_name, record)
            return True
        else:
            logger.info("🛡️ Champion Defends.")
            try:
                shutil.rmtree(candidate_path)
                logger.info("🗑️ Discarded challenger model.")
            except: pass
            return False

    def _cleanup_archive(self, archive_path, keep=3):
        if not os.path.exists(archive_path): return
        archives = sorted([os.path.join(archive_path, d) for d in os.listdir(archive_path)], key=os.path.getmtime)
        while len(archives) > keep:
            target = archives.pop(0)
            shutil.rmtree(target)