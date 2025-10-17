import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Try to import optional libraries
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class ModelTrainer:
    """
    Trains and evaluates multiple regression models for real estate price prediction.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.results = {}

    def train_multiple_models(self, X: pd.DataFrame, y: pd.Series,
                             test_size: float = 0.2,
                             random_state: int = 42) -> Dict[str, Dict[str, float]]:
        """
        Train multiple regression models and compare their performance.

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility

        Returns:
            Dictionary with model names as keys and metrics as values
        """
        try:
            self.logger.info(f"Training models with {len(X)} samples and {len(X.columns)} features")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Define models to train
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=random_state,
                    n_jobs=-1
                )
            }

            # Add XGBoost if available
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=random_state,
                    n_jobs=-1
                )

            # Add LightGBM if available
            if LIGHTGBM_AVAILABLE:
                models['LightGBM'] = LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1
                )

            results = {}

            # Train and evaluate each model
            for name, model in models.items():
                self.logger.info(f"Training {name}...")

                try:
                    # Train model
                    model.fit(X_train, y_train)

                    # Make predictions
                    y_pred = model.predict(X_test)

                    # Calculate metrics
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                    # Cross-validation score
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=5,
                        scoring='r2',
                        n_jobs=-1
                    )
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()

                    results[name] = {
                        'r2': r2,
                        'mae': mae,
                        'rmse': rmse,
                        'cv_mean': cv_mean,
                        'cv_std': cv_std,
                        'model': model
                    }

                    self.logger.info(
                        f"{name} - R²: {r2:.4f}, MAE: {mae:.2f}, "
                        f"RMSE: {rmse:.2f}, CV: {cv_mean:.4f} (+/- {cv_std:.4f})"
                    )

                except Exception as e:
                    self.logger.error(f"Error training {name}: {str(e)}")
                    continue

            if not results:
                self.logger.error("No models were successfully trained")
                return None

            # Store models and results
            self.models = {name: res['model'] for name, res in results.items()}
            self.results = results

            return results

        except Exception as e:
            self.logger.error(f"Error in train_multiple_models: {str(e)}")
            return None

    def get_best_model(self, metric: str = 'r2') -> tuple[str, Any]:
        """
        Get the best performing model based on specified metric.

        Args:
            metric: Metric to use for comparison ('r2', 'mae', 'rmse')

        Returns:
            Tuple of (model_name, model_object)
        """
        if not self.results:
            return None, None

        if metric in ['mae', 'rmse']:
            # Lower is better
            best_name = min(self.results.keys(), key=lambda k: self.results[k][metric])
        else:
            # Higher is better (r2)
            best_name = max(self.results.keys(), key=lambda k: self.results[k][metric])

        return best_name, self.models[best_name]

    def save_model(self, model_name: str, filepath: str) -> bool:
        """
        Save a trained model to disk.

        Args:
            model_name: Name of the model to save
            filepath: Path where to save the model

        Returns:
            True if successful, False otherwise
        """
        try:
            if model_name not in self.models:
                self.logger.error(f"Model {model_name} not found")
                return False

            joblib.dump(self.models[model_name], filepath)
            self.logger.info(f"Model {model_name} saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self, filepath: str) -> Any:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded model object or None if failed
        """
        try:
            model = joblib.load(filepath)
            self.logger.info(f"Model loaded from {filepath}")
            return model

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None
