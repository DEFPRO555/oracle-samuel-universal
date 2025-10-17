"""
MD5 Manager for Oracle Samuel Application

This module provides functions for generating MD5 hashes from DataFrames
and creating signature records for data integrity tracking.
"""

import hashlib
import pandas as pd
from datetime import datetime
from typing import Dict, Any


def generate_md5_from_dataframe(df: pd.DataFrame) -> str:
    """
    Generate MD5 hash from a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to generate hash from
        
    Returns:
        str: MD5 hash string
    """
    # Convert DataFrame to string representation for hashing
    # This includes data, index, and column information
    df_string = df.to_string()
    
    # Generate MD5 hash
    md5_hash = hashlib.md5(df_string.encode('utf-8')).hexdigest()
    
    return md5_hash


def create_signature_record(filename: str, md5_hash: str) -> Dict[str, Any]:
    """
    Create a signature record for data integrity tracking.
    
    Args:
        filename (str): Name of the file or data source
        md5_hash (str): MD5 hash of the data
        
    Returns:
        Dict[str, Any]: Signature record dictionary
    """
    signature_record = {
        'filename': filename,
        'md5_hash': md5_hash,
        'timestamp': datetime.now().isoformat(),
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return signature_record
