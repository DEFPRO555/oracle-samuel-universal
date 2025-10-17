"""
Self-learning module for Oracle Samuel Real Estate Prediction System.
This module provides automated model training, evaluation, and improvement capabilities.
"""

__version__ = "1.0.0"
__author__ = "Oracle Samuel Team"

from .trainer import SelfLearningTrainer
from .evaluator import ModelEvaluator
from .retrain_manager import RetrainManager
from .feedback_manager import FeedbackManager
from .knowledge_base import KnowledgeBase

__all__ = [
    'SelfLearningTrainer',
    'ModelEvaluator', 
    'RetrainManager',
    'FeedbackManager',
    'KnowledgeBase'
]
