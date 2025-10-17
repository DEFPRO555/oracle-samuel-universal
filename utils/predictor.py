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
        self.model = None  # Alias for best_model for backwards compatibility
        self.feature_importance = None  # Store feature importance
        self.metrics = {}  # Store metrics in expected format
        
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
        self.model = self.best_model  # Alias for backwards compatibility

        # Store feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(self.best_model, 'coef_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': np.abs(self.best_model.coef_)
            }).sort_values('importance', ascending=False)

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

        # Set self.metrics to best model's metrics for backwards compatibility
        if self.best_model:
            for name, model in self.models.items():
                if model is self.best_model and name in evaluation_results:
                    self.metrics = {
                        'r2_score': evaluation_results[name]['r2'],
                        'mae': evaluation_results[name]['mae'],
                        'rmse': evaluation_results[name]['rmse'],
                        'mse': evaluation_results[name]['mse']
                    }
                    break

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

    def train_model(self, model_type: str = 'linear_regression') -> Tuple[bool, Dict[str, float], np.ndarray, np.ndarray]:
        """
        Train a specific model type on the data stored in session state.

        Args:
            model_type: Type of model to train

        Returns:
            Tuple of (success, metrics, y_test, y_pred)
        """
        try:
            import streamlit as st

            # Check if cleaned data exists
            if 'cleaned_df' not in st.session_state or st.session_state.cleaned_df is None:
                return False, {'error': 'No cleaned data available'}, np.array([]), np.array([])

            cleaned_df = st.session_state.cleaned_df

            # Find target column
            target_column = None
            possible_targets = ['price', 'Price', 'PRICE', 'SalePrice', 'sale_price']
            for col in possible_targets:
                if col in cleaned_df.columns:
                    target_column = col
                    break

            if target_column is None:
                return False, {'error': 'No target column found'}, np.array([]), np.array([])

            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(cleaned_df, target_column)

            # Train models
            train_scores = self.train_models(X_train, y_train)

            # Evaluate on test set
            eval_results = self.evaluate_models(X_test, y_test)

            # Get predictions from best model
            y_pred = self.predict(X_test)

            # Get metrics for best model
            best_model_name = None
            for name, model in self.models.items():
                if model is self.best_model:
                    best_model_name = name
                    break

            if best_model_name and best_model_name in eval_results:
                metrics = eval_results[best_model_name]
            else:
                # Use first model's metrics if best model not found
                metrics = list(eval_results.values())[0] if eval_results else {}

            return True, metrics, y_test.values, y_pred

        except Exception as e:
            self.logger.error(f"Error in train_model: {str(e)}")
            return False, {'error': str(e)}, np.array([]), np.array([])

    def analyze_correlation(self) -> Optional[pd.Series]:
        """
        Analyze correlations between features and target column.

        Returns:
            Series with correlation values, sorted by absolute value
        """
        try:
            import streamlit as st

            # Check if cleaned data exists
            if 'cleaned_df' not in st.session_state or st.session_state.cleaned_df is None:
                self.logger.warning("No cleaned data available for correlation analysis")
                return None

            cleaned_df = st.session_state.cleaned_df

            # Find target column
            target_column = None
            possible_targets = ['price', 'Price', 'PRICE', 'SalePrice', 'sale_price']
            for col in possible_targets:
                if col in cleaned_df.columns:
                    target_column = col
                    break

            if target_column is None:
                self.logger.warning("No target column found for correlation analysis")
                return None

            # Select only numeric columns
            numeric_df = cleaned_df.select_dtypes(include=[np.number])

            if target_column not in numeric_df.columns:
                self.logger.warning(f"Target column {target_column} is not numeric")
                return None

            # Calculate correlations with target
            correlations = numeric_df.corr()[target_column]

            # Sort by absolute value, descending
            correlations_sorted = correlations.abs().sort_values(ascending=False)

            # Return the correlations in the sorted order (with original signs)
            return correlations[correlations_sorted.index]

        except Exception as e:
            self.logger.error(f"Error in analyze_correlation: {str(e)}")
            return None

    def get_top_features(self, n: int = 5) -> pd.DataFrame:
        """
        Get top N most important features.

        Args:
            n: Number of top features to return

        Returns:
            DataFrame with top features and their importance
        """
        if self.feature_importance is not None:
            return self.feature_importance.head(n)
        else:
            self.logger.warning("Feature importance not available. Train a model first.")
            return pd.DataFrame(columns=['feature', 'importance'])
