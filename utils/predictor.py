import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging
from typing import Dict, Any, Tuple, Optional, List
import json
import os

class RealEstatePredictor:
    """
    Advanced real estate price prediction system with multiple ML models.
    Supports linear regression, ensemble methods, and model comparison.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.target_column = None
        self.best_model = None
        self.model_metrics = {}
        
    def prepare_data(self, df: pd.DataFrame, target_column: str, 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training by encoding categorical variables and scaling features.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        self.logger.info(f"Preparing data for prediction with target: {target_column}")
        self.target_column = target_column
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature columns
        self.feature_columns = list(X.columns)
        
        # Handle categorical variables
        X_encoded = self._encode_categorical_features(X)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        # Convert back to DataFrame
        X_train_df = pd.DataFrame(X_train_scaled, columns=X_encoded.columns, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_scaled, columns=X_encoded.columns, index=X_test.index)
        
        self.logger.info(f"Data prepared. Train shape: {X_train_df.shape}, Test shape: {X_test_df.shape}")
        
        return X_train_df, X_test_df, y_train, y_test
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using label encoding."""
        X_encoded = X.copy()
        
        for column in X_encoded.columns:
            if X_encoded[column].dtype == 'object' or X_encoded[column].dtype.name == 'string':
                # Use label encoding for categorical variables
                encoder = LabelEncoder()
                X_encoded[column] = encoder.fit_transform(X_encoded[column].astype(str))
                self.encoders[column] = encoder
        
        return X_encoded
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train multiple regression models and compare their performance.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary with model performance metrics
        """
        self.logger.info("Training multiple regression models")
        
        # Define models to train
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        model_scores = {}
        
        for name, model in models_to_train.items():
            self.logger.info(f"Training {name}")
            
            # Train the model
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            model_scores[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
        
        # Select best model based on cross-validation score
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['cv_mean'])
        self.best_model = self.models[best_model_name]
        
        self.logger.info(f"Best model: {best_model_name} with CV score: {model_scores[best_model_name]['cv_mean']:.4f}")
        
        return model_scores
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate all trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation metrics for all models
        """
        self.logger.info("Evaluating models on test data")
        
        evaluation_results = {}
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            evaluation_results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred.tolist()
            }
        
        self.model_metrics = evaluation_results
        return evaluation_results
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using the specified model or best model.
        
        Args:
            X: Features for prediction
            model_name: Name of model to use (if None, uses best model)
            
        Returns:
            Predictions array
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No model available for prediction. Train a model first.")
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
            model = self.models[model_name]
        
        # Encode categorical features if needed
        X_encoded = self._encode_categorical_features(X)
        
        # Scale features if scaler is available
        if 'main' in self.scalers:
            X_scaled = self.scalers['main'].transform(X_encoded)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)
        else:
            X_scaled_df = X_encoded
        
        # Make predictions
        predictions = model.predict(X_scaled_df)
        
        return predictions
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> Dict[str, float]:
        """
        Get feature importance from the model.
        
        Args:
            model_name: Name of model to get importance from
            
        Returns:
            Dictionary with feature importance scores
        """
        if model_name is None:
            model = self.best_model
            model_name = "Best Model"
        else:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found")
            model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importance_dict = dict(zip(self.feature_columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # For linear models
            importance_dict = dict(zip(self.feature_columns, abs(model.coef_)))
        else:
            self.logger.warning(f"Model {model_name} does not support feature importance")
            return {}
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def save_model(self, filepath: str, model_name: Optional[str] = None):
        """
        Save the trained model and related objects.
        
        Args:
            filepath: Path to save the model
            model_name: Name of model to save (if None, saves best model)
        """
        if model_name is None:
            model = self.best_model
            model_name = "best_model"
        else:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found")
            model = self.models[model_name]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        model_path = f"{filepath}_{model_name}.pkl"
        joblib.dump(model, model_path)
        
        # Save scaler and encoders
        if 'main' in self.scalers:
            scaler_path = f"{filepath}_scaler.pkl"
            joblib.dump(self.scalers['main'], scaler_path)
        
        if self.encoders:
            encoders_path = f"{filepath}_encoders.pkl"
            joblib.dump(self.encoders, encoders_path)
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_name': model_name,
            'model_metrics': self.model_metrics
        }
        
        metadata_path = f"{filepath}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, filepath: str):
        """
        Load a previously saved model and related objects.
        
        Args:
            filepath: Base path of the saved model files
        """
        # Load metadata
        metadata_path = f"{filepath}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_columns = metadata['feature_columns']
            self.target_column = metadata['target_column']
            self.model_metrics = metadata.get('model_metrics', {})
        
        # Load model files
        model_files = [f for f in os.listdir(os.path.dirname(filepath)) if f.startswith(os.path.basename(filepath))]
        
        for file in model_files:
            full_path = os.path.join(os.path.dirname(filepath), file)
            
            if file.endswith('_scaler.pkl'):
                self.scalers['main'] = joblib.load(full_path)
            elif file.endswith('_encoders.pkl'):
                self.encoders = joblib.load(full_path)
            elif file.endswith('.pkl') and not file.endswith('_scaler.pkl') and not file.endswith('_encoders.pkl'):
                model_name = file.replace(f"{os.path.basename(filepath)}_", "").replace('.pkl', '')
                self.models[model_name] = joblib.load(full_path)
                if model_name == 'best_model':
                    self.best_model = self.models[model_name]
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all trained models and their performance.
        
        Returns:
            Dictionary with model summary information
        """
        summary = {
            'total_models': len(self.models),
            'best_model': None,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_performance': self.model_metrics
        }
        
        if self.best_model:
            # Find the name of the best model
            for name, model in self.models.items():
                if model is self.best_model:
                    summary['best_model'] = name
                    break
        
        return summary
