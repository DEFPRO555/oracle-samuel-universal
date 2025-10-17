"""
Database Manager for Oracle Samuel Application

This module provides database operations for storing and retrieving
data, signatures, and model metrics.
"""

import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import pickle


class DatabaseManager:
    """
    Database manager for handling data storage and retrieval operations.
    Uses local file storage for simplicity in Streamlit Cloud deployment.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the database manager.
        
        Args:
            data_dir (str): Directory to store data files
        """
        self.data_dir = data_dir
        self.signatures_file = os.path.join(data_dir, "signatures.json")
        self.model_metrics_file = os.path.join(data_dir, "model_metrics.json")
        self.uploaded_data_file = os.path.join(data_dir, "uploaded_data.pkl")
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize files if they don't exist
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize data files if they don't exist."""
        if not os.path.exists(self.signatures_file):
            with open(self.signatures_file, 'w') as f:
                json.dump([], f)
        
        if not os.path.exists(self.model_metrics_file):
            with open(self.model_metrics_file, 'w') as f:
                json.dump([], f)
    
    def save_uploaded_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Save uploaded DataFrame to storage.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            
        Returns:
            Tuple[bool, str]: Success status and message
        """
        try:
            # Save DataFrame as pickle for efficiency
            with open(self.uploaded_data_file, 'wb') as f:
                pickle.dump(df, f)
            
            return True, "Data saved successfully"
        except Exception as e:
            return False, f"Error saving data: {str(e)}"
    
    def load_uploaded_data(self) -> Optional[pd.DataFrame]:
        """
        Load the most recently uploaded DataFrame.
        
        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None if not found
        """
        try:
            if os.path.exists(self.uploaded_data_file):
                with open(self.uploaded_data_file, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def save_signature(self, signature: Dict[str, Any]) -> bool:
        """
        Save a signature record.
        
        Args:
            signature (Dict[str, Any]): Signature record to save
            
        Returns:
            bool: Success status
        """
        try:
            # Load existing signatures
            signatures = []
            if os.path.exists(self.signatures_file):
                with open(self.signatures_file, 'r') as f:
                    signatures = json.load(f)
            
            # Add new signature
            signatures.append(signature)
            
            # Save back to file
            with open(self.signatures_file, 'w') as f:
                json.dump(signatures, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving signature: {str(e)}")
            return False
    
    def get_signatures(self) -> list:
        """
        Get all signature records.
        
        Returns:
            list: List of signature records
        """
        try:
            if os.path.exists(self.signatures_file):
                with open(self.signatures_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading signatures: {str(e)}")
            return []
    
    def save_model_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Save model metrics.
        
        Args:
            metrics (Dict[str, Any]): Model metrics to save
            
        Returns:
            bool: Success status
        """
        try:
            # Add timestamp to metrics
            metrics['timestamp'] = datetime.now().isoformat()
            metrics['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Load existing metrics
            all_metrics = []
            if os.path.exists(self.model_metrics_file):
                with open(self.model_metrics_file, 'r') as f:
                    all_metrics = json.load(f)
            
            # Add new metrics
            all_metrics.append(metrics)
            
            # Save back to file
            with open(self.model_metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving model metrics: {str(e)}")
            return False
    
    def get_model_metrics(self) -> pd.DataFrame:
        """
        Get all model metrics as a DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing all model metrics
        """
        try:
            if os.path.exists(self.model_metrics_file):
                with open(self.model_metrics_file, 'r') as f:
                    metrics_list = json.load(f)
                
                if metrics_list:
                    return pd.DataFrame(metrics_list)
                else:
                    return pd.DataFrame()
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading model metrics: {str(e)}")
            return pd.DataFrame()
    
    def clear_all_data(self) -> bool:
        """
        Clear all stored data.
        
        Returns:
            bool: Success status
        """
        try:
            # Clear signatures
            with open(self.signatures_file, 'w') as f:
                json.dump([], f)
            
            # Clear model metrics
            with open(self.model_metrics_file, 'w') as f:
                json.dump([], f)
            
            # Remove uploaded data file
            if os.path.exists(self.uploaded_data_file):
                os.remove(self.uploaded_data_file)
            
            return True
        except Exception as e:
            print(f"Error clearing data: {str(e)}")
            return False
