import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

class ModelEvaluator:
    """Advanced model evaluation and performance monitoring system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.evaluation_history = []
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str = "Model") -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        evaluation_record = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'metrics': metrics
        }
        
        self.evaluation_history.append(evaluation_record)
        return metrics
    
    def get_evaluation_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get evaluation history."""
        if limit:
            return self.evaluation_history[-limit:]
        return self.evaluation_history
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trends analysis."""
        if not self.evaluation_history:
            return {'message': 'No evaluation history available'}
        
        # Calculate trends
        r2_scores = [eval_record['metrics']['r2'] for eval_record in self.evaluation_history]
        mae_scores = [eval_record['metrics']['mae'] for eval_record in self.evaluation_history]
        
        return {
            'r2_trend': 'improving' if len(r2_scores) > 1 and r2_scores[-1] > r2_scores[0] else 'stable',
            'mae_trend': 'improving' if len(mae_scores) > 1 and mae_scores[-1] < mae_scores[0] else 'stable',
            'latest_r2': r2_scores[-1] if r2_scores else 0,
            'latest_mae': mae_scores[-1] if mae_scores else 0,
            'total_evaluations': len(self.evaluation_history)
        }
