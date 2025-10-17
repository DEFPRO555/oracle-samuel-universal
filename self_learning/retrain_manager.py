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

    def retrain_all(self) -> tuple[bool, Dict[str, Any]]:
        """
        Retrain all models and return the best performing model.

        Returns:
            Tuple of (success: bool, result: Dict with model metrics)
        """
        try:
            self.logger.info("Starting model retraining...")

            # Import required modules
            import streamlit as st
            from models.model_trainer import ModelTrainer

            # Check if cleaned data exists
            if 'cleaned_df' not in st.session_state or st.session_state.cleaned_df is None:
                return False, "No cleaned data available. Please upload and clean data first."

            cleaned_df = st.session_state.cleaned_df

            # Check if target column exists (common real estate price columns)
            target_column = None
            possible_targets = ['price', 'Price', 'PRICE', 'SalePrice', 'sale_price']
            for col in possible_targets:
                if col in cleaned_df.columns:
                    target_column = col
                    break

            if target_column is None:
                return False, "No target column found. Please ensure your data has a 'price' column."

            # Prepare features and target
            X = cleaned_df.drop(columns=[target_column])
            y = cleaned_df[target_column]

            # Select only numeric columns for features
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_cols]

            if X.empty or len(numeric_cols) == 0:
                return False, "No numeric features found in the dataset."

            # Initialize trainer and train models
            trainer = ModelTrainer()
            results = trainer.train_multiple_models(X, y)

            if results is None or len(results) == 0:
                return False, "Model training failed. No results returned."

            # Find best model
            best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
            best_result = results[best_model_name]

            # Record in history
            retrain_record = {
                'timestamp': datetime.now().isoformat(),
                'model_name': best_model_name,
                'reason': 'Manual retrain triggered',
                'r2': best_result['r2'],
                'mae': best_result['mae'],
                'rmse': best_result['rmse'],
                'success': True
            }
            self.retrain_history.append(retrain_record)

            # Update session state with best model
            st.session_state.best_model = best_model_name
            st.session_state.model_results = results

            self.logger.info(f"Retraining complete. Best model: {best_model_name} with R²={best_result['r2']:.4f}")

            return True, {
                'best_model': best_model_name,
                'r2': best_result['r2'],
                'mae': best_result['mae'],
                'rmse': best_result['rmse'],
                'all_results': results
            }

        except Exception as e:
            self.logger.error(f"Error during retraining: {str(e)}")
            return False, f"Retraining error: {str(e)}"
