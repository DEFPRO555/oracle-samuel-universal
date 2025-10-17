import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SelfLearningTrainer:
    """
    Advanced self-learning trainer for real estate prediction models.
    Automatically selects best algorithms and hyperparameters.
    """
    
    def __init__(self, model_save_path: str = "data/models"):
        self.model_save_path = model_save_path
        self.logger = logging.getLogger(__name__)
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.training_history = []
        self.feature_importance = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(model_save_path, exist_ok=True)
    
    def auto_train(self, df: pd.DataFrame, target_column: str, 
                   test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Automatically train multiple models and select the best one.
        
        Args:
            df: Training DataFrame
            target_column: Name of target column
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Training results dictionary
        """
        self.logger.info(f"Starting auto-training with target: {target_column}")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self._prepare_data(df, target_column, test_size, random_state)
        
        # Train multiple models
        model_results = self._train_multiple_models(X_train, y_train, X_test, y_test)
        
        # Select best model
        best_model_info = self._select_best_model(model_results)
        
        # Save training history
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'target_column': target_column,
            'data_shape': df.shape,
            'best_model': best_model_info,
            'all_results': model_results
        }
        self.training_history.append(training_record)
        
        # Save models and results
        self._save_training_results()
        
        return {
            'best_model': best_model_info,
            'all_models': model_results,
            'training_summary': self._get_training_summary()
        }
    
    def _prepare_data(self, df: pd.DataFrame, target_column: str, 
                     test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for training."""
        # Remove rows with missing target values
        df_clean = df.dropna(subset=[target_column])
        
        # Separate features and target
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]
        
        # Handle categorical variables
        X_encoded = self._encode_categorical_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_df = pd.DataFrame(X_train_scaled, columns=X_encoded.columns, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_scaled, columns=X_encoded.columns, index=X_test.index)
        
        return X_train_df, X_test_df, y_train, y_test
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        X_encoded = X.copy()
        self.encoders = {}
        
        for column in X_encoded.columns:
            if X_encoded[column].dtype == 'object' or X_encoded[column].dtype.name == 'string':
                encoder = LabelEncoder()
                X_encoded[column] = encoder.fit_transform(X_encoded[column].astype(str))
                self.encoders[column] = encoder
        
        return X_encoded
    
    def _train_multiple_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train multiple models and evaluate them."""
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            self.logger.info(f"Training {name}")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_metrics = self._calculate_metrics(y_train, y_pred_train)
                test_metrics = self._calculate_metrics(y_test, y_pred_test)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                results[name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist()
                }
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(X_train.columns, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    self.feature_importance[name] = dict(zip(X_train.columns, abs(model.coef_)))
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        return results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def _select_best_model(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best model based on cross-validation score."""
        valid_models = {name: result for name, result in model_results.items() 
                       if 'error' not in result}
        
        if not valid_models:
            raise ValueError("No models were successfully trained")
        
        # Select best model based on CV score
        best_name = max(valid_models.keys(), 
                       key=lambda x: valid_models[x]['cv_mean'])
        
        self.best_model = self.trained_models[best_name]
        self.best_model_name = best_name
        
        return {
            'name': best_name,
            'model': self.best_model,
            'metrics': valid_models[best_name],
            'feature_importance': self.feature_importance.get(best_name, {})
        }
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                             model_name: str = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for the specified model.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name of model to tune (if None, uses best model)
            
        Returns:
            Tuning results
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found")
        
        # Define parameter grids for different models
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'Ridge Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        }
        
        if model_name not in param_grids:
            self.logger.warning(f"No parameter grid defined for {model_name}")
            return {'error': f'No parameter grid for {model_name}'}
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.trained_models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.trained_models[model_name] = grid_search.best_estimator_
        if model_name == self.best_model_name:
            self.best_model = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        Make predictions using the specified model.
        
        Args:
            X: Features for prediction
            model_name: Name of model to use (if None, uses best model)
            
        Returns:
            Predictions array
        """
        if model_name is None:
            model = self.best_model
        else:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} not found")
            model = self.trained_models[model_name]
        
        # Encode categorical features
        X_encoded = self._encode_categorical_features(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_encoded)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        return predictions
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance summary of all trained models."""
        if not self.trained_models:
            return {'message': 'No models trained yet'}
        
        performance = {}
        for name, model in self.trained_models.items():
            if name in self.feature_importance:
                performance[name] = {
                    'feature_importance': self.feature_importance[name],
                    'model_type': type(model).__name__
                }
        
        return {
            'best_model': self.best_model_name,
            'total_models': len(self.trained_models),
            'model_performance': performance,
            'training_history_count': len(self.training_history)
        }
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training session."""
        return {
            'models_trained': len(self.trained_models),
            'best_model': self.best_model_name,
            'feature_count': len(self.feature_importance.get(self.best_model_name, {})),
            'training_timestamp': datetime.now().isoformat()
        }
    
    def _save_training_results(self):
        """Save training results to disk."""
        # Save best model
        if self.best_model is not None:
            model_path = os.path.join(self.model_save_path, f"{self.best_model_name}_best.pkl")
            joblib.dump(self.best_model, model_path)
        
        # Save scaler
        if hasattr(self, 'scaler'):
            scaler_path = os.path.join(self.model_save_path, "scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
        
        # Save encoders
        if hasattr(self, 'encoders'):
            encoders_path = os.path.join(self.model_save_path, "encoders.pkl")
            joblib.dump(self.encoders, encoders_path)
        
        # Save training history
        history_path = os.path.join(self.model_save_path, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        # Save feature importance
        if self.feature_importance:
            importance_path = os.path.join(self.model_save_path, "feature_importance.json")
            with open(importance_path, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
        
        self.logger.info(f"Training results saved to {self.model_save_path}")
    
    def load_trained_model(self, model_path: str):
        """
        Load a previously trained model.
        
        Args:
            model_path: Path to the model directory
        """
        # Load best model
        best_model_files = [f for f in os.listdir(model_path) if f.endswith('_best.pkl')]
        if best_model_files:
            model_file = os.path.join(model_path, best_model_files[0])
            self.best_model = joblib.load(model_file)
            self.best_model_name = best_model_files[0].replace('_best.pkl', '')
        
        # Load scaler
        scaler_path = os.path.join(model_path, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        # Load encoders
        encoders_path = os.path.join(model_path, "encoders.pkl")
        if os.path.exists(encoders_path):
            self.encoders = joblib.load(encoders_path)
        
        # Load training history
        history_path = os.path.join(model_path, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        
        # Load feature importance
        importance_path = os.path.join(model_path, "feature_importance.json")
        if os.path.exists(importance_path):
            with open(importance_path, 'r') as f:
                self.feature_importance = json.load(f)
        
        self.logger.info(f"Model loaded from {model_path}")
    
    def retrain_with_new_data(self, new_df: pd.DataFrame, target_column: str,
                             retrain_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Retrain model with new data if performance degradation is detected.
        
        Args:
            new_df: New data for retraining
            target_column: Name of target column
            retrain_threshold: Threshold for triggering retraining
            
        Returns:
            Retraining results
        """
        if self.best_model is None:
            return {'error': 'No trained model available for retraining'}
        
        # Evaluate current model on new data
        X_new = new_df.drop(columns=[target_column])
        y_new = new_df[target_column]
        
        # Make predictions with current model
        y_pred = self.predict(X_new)
        
        # Calculate performance
        current_performance = self._calculate_metrics(y_new, y_pred)
        
        # Check if retraining is needed
        if current_performance['r2'] < (1 - retrain_threshold):
            self.logger.info("Performance degradation detected, retraining model")
            
            # Retrain with combined data
            combined_df = pd.concat([new_df], ignore_index=True)
            retrain_results = self.auto_train(combined_df, target_column)
            
            return {
                'retrained': True,
                'previous_performance': current_performance,
                'retrain_results': retrain_results
            }
        else:
            return {
                'retrained': False,
                'current_performance': current_performance,
                'message': 'Model performance is acceptable, no retraining needed'
            }
