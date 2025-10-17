import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json

class FeedbackManager:
    """Manages user feedback and model improvement suggestions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feedback_history = []
    
    def collect_feedback(self, prediction: float, actual: float, 
                        user_rating: int, comments: str = "") -> Dict[str, Any]:
        """Collect user feedback on predictions."""
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'actual': actual,
            'user_rating': user_rating,
            'comments': comments,
            'prediction_error': abs(prediction - actual)
        }
        
        self.feedback_history.append(feedback)
        return feedback
    
    def analyze_feedback_trends(self) -> Dict[str, Any]:
        """Analyze feedback trends to identify improvement areas."""
        if not self.feedback_history:
            return {'message': 'No feedback available'}
        
        avg_rating = np.mean([f['user_rating'] for f in self.feedback_history])
        avg_error = np.mean([f['prediction_error'] for f in self.feedback_history])
        
        return {
            'total_feedback': len(self.feedback_history),
            'average_rating': avg_rating,
            'average_error': avg_error,
            'feedback_trend': 'improving' if avg_rating > 3 else 'needs_improvement'
        }
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get a summary of all feedback received."""
        if not self.feedback_history:
            return {
                'total_feedback': 0,
                'average_rating': 0,
                'average_error': 0,
                'feedback_trend': 'no_data'
            }
        
        avg_rating = np.mean([f['user_rating'] for f in self.feedback_history])
        avg_error = np.mean([f['prediction_error'] for f in self.feedback_history])
        
        return {
            'total_feedback': len(self.feedback_history),
            'average_rating': avg_rating,
            'average_error': avg_error,
            'feedback_trend': 'improving' if avg_rating > 3 else 'needs_improvement',
            'recent_feedback': self.feedback_history[-5:] if len(self.feedback_history) > 5 else self.feedback_history
        }
    
    def get_all_feedback(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get all feedback with optional limit."""
        if limit:
            return self.feedback_history[-limit:]
        return self.feedback_history
