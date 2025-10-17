import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta

class RetrainManager:
    """Manages automatic model retraining based on performance and data drift."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.retrain_history = []
    
    def should_retrain(self, performance_metrics: Dict[str, float], 
                      threshold: float = 0.1) -> bool:
        """Determine if model should be retrained based on performance."""
        if performance_metrics.get('r2', 0) < (1 - threshold):
            return True
        return False
    
    def schedule_retrain(self, model_name: str, reason: str) -> Dict[str, Any]:
        """Schedule a model retraining."""
        retrain_record = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'reason': reason,
            'scheduled': True
        }
        self.retrain_history.append(retrain_record)
        return retrain_record

    def get_retrain_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """Returns the retrain history."""
        self.logger.info(f"Retrieving retrain history (limit: {limit})")
        if limit:
            return self.retrain_history[-limit:]
        return self.retrain_history
