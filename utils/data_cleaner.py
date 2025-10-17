import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

class DataCleaner:
    """
    A comprehensive data cleaning utility for real estate data.
    Handles missing values, outliers, data type conversions, and data validation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cleaning_stats = {}
    
    def clean_dataframe(self, df: pd.DataFrame, 
                       target_column: Optional[str] = None,
                       remove_outliers: bool = True,
                       fill_missing: bool = True) -> pd.DataFrame:
        """
        Comprehensive data cleaning pipeline.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column for regression
            remove_outliers: Whether to remove outliers
            fill_missing: Whether to fill missing values
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info(f"Starting data cleaning for DataFrame with shape {df.shape}")
        original_shape = df.shape
        
        # Create a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Step 1: Handle missing values
        if fill_missing:
            cleaned_df = self._handle_missing_values(cleaned_df)
        
        # Step 2: Convert data types
        cleaned_df = self._convert_data_types(cleaned_df)
        
        # Step 3: Remove outliers if requested
        if remove_outliers and target_column:
            cleaned_df = self._remove_outliers(cleaned_df, target_column)
        
        # Step 4: Validate data
        cleaned_df = self._validate_data(cleaned_df)
        
        # Record cleaning statistics
        self.cleaning_stats = {
            'original_shape': original_shape,
            'cleaned_shape': cleaned_df.shape,
            'rows_removed': original_shape[0] - cleaned_df.shape[0],
            'columns_processed': len(cleaned_df.columns)
        }
        
        self.logger.info(f"Data cleaning completed. Shape: {original_shape} -> {cleaned_df.shape}")
        return cleaned_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        missing_before = df.isnull().sum().sum()
        
        for column in df.columns:
            if df[column].dtype in ['object', 'string']:
                # For categorical data, fill with mode or 'Unknown'
                mode_value = df[column].mode()
                if not mode_value.empty:
                    df[column].fillna(mode_value[0], inplace=True)
                else:
                    df[column].fillna('Unknown', inplace=True)
            else:
                # For numerical data, fill with median
                df[column].fillna(df[column].median(), inplace=True)
        
        missing_after = df.isnull().sum().sum()
        self.logger.info(f"Missing values handled: {missing_before} -> {missing_after}")
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types to appropriate formats."""
        for column in df.columns:
            # Try to convert to numeric if possible
            if df[column].dtype == 'object':
                try:
                    # Try to convert to numeric
                    numeric_series = pd.to_numeric(df[column], errors='coerce')
                    if not numeric_series.isna().all():
                        df[column] = numeric_series
                except:
                    # Keep as string if conversion fails
                    df[column] = df[column].astype('string')
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        if target_column not in df.columns:
            return df
        
        original_count = len(df)
        
        # Calculate IQR for target column
        Q1 = df[target_column].quantile(0.25)
        Q3 = df[target_column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove outliers
        df_cleaned = df[(df[target_column] >= lower_bound) & 
                       (df[target_column] <= upper_bound)]
        
        outliers_removed = original_count - len(df_cleaned)
        self.logger.info(f"Outliers removed: {outliers_removed}")
        
        return df_cleaned
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data integrity."""
        # Remove duplicate rows
        duplicates_before = df.duplicated().sum()
        df = df.drop_duplicates()
        duplicates_after = df.duplicated().sum()
        
        if duplicates_before > 0:
            self.logger.info(f"Duplicates removed: {duplicates_before}")
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        return df
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get a report of the cleaning operations performed."""
        return {
            'cleaning_stats': self.cleaning_stats,
            'summary': f"Cleaned data from {self.cleaning_stats.get('original_shape', 'Unknown')} to {self.cleaning_stats.get('cleaned_shape', 'Unknown')}"
        }
    
    def validate_uploaded_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate uploaded data for real estate prediction.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_info': {}
        }
        
        # Check if DataFrame is empty
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Uploaded data is empty")
            return validation_results
        
        # Check minimum number of rows
        if len(df) < 10:
            validation_results['warnings'].append("Dataset has fewer than 10 rows, which may affect model performance")
        
        # Check for required columns (basic real estate features)
        expected_columns = ['price', 'area', 'bedrooms', 'bathrooms']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['warnings'].append(f"Missing common real estate columns: {missing_columns}")
        
        # Check data types
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) < 2:
            validation_results['warnings'].append("Dataset has very few numeric columns")
        
        # Check for excessive missing values
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        high_missing_cols = missing_percentage[missing_percentage > 50].index.tolist()
        
        if high_missing_cols:
            validation_results['warnings'].append(f"Columns with >50% missing values: {high_missing_cols}")
        
        # Store data information
        validation_results['data_info'] = {
            'shape': df.shape,
            'columns': list(df.columns),
            'numeric_columns': list(numeric_columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        return validation_results
