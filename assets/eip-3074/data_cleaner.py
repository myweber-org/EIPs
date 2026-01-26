
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(file_path: str, 
                   missing_strategy: str = 'drop',
                   fill_value: Optional[float] = None) -> pd.DataFrame:
    """
    Load and clean CSV data by handling missing values.
    
    Args:
        file_path: Path to CSV file
        missing_strategy: Strategy for handling missing values 
                         ('drop', 'fill', 'mean')
        fill_value: Value to fill when using 'fill' strategy
    
    Returns:
        Cleaned pandas DataFrame
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if df.empty:
        return df
    
    if missing_strategy == 'drop':
        df_cleaned = df.dropna()
    elif missing_strategy == 'fill':
        if fill_value is None:
            raise ValueError("fill_value must be provided for 'fill' strategy")
        df_cleaned = df.fillna(fill_value)
    elif missing_strategy == 'mean':
        df_cleaned = df.fillna(df.mean(numeric_only=True))
    else:
        raise ValueError(f"Unknown strategy: {missing_strategy}")
    
    return df_cleaned

def remove_outliers_iqr(df: pd.DataFrame, 
                       columns: Optional[list] = None,
                       multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using Interquartile Range method.
    
    Args:
        df: Input DataFrame
        columns: Columns to process (None for all numeric columns)
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filtered = df.copy()
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_filtered = df_filtered[mask]
    
    return df_filtered

def normalize_data(df: pd.DataFrame,
                  columns: Optional[list] = None,
                  method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize numeric columns in DataFrame.
    
    Args:
        df: Input DataFrame
        columns: Columns to normalize (None for all numeric columns)
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val != 0:
                    df_normalized[col] = (df[col] - mean_val) / std_val
            else:
                raise ValueError(f"Unknown normalization method: {method}")
    
    return df_normalized

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame for common data quality issues.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if DataFrame passes validation checks
    """
    if df.empty:
        return False
    
    if df.isnull().all().any():
        return False
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            if df[col].isnull().all():
                return False
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, 20, 30, np.nan, 50],
        'C': [100, 200, 300, 400, 500]
    })
    
    cleaned = clean_csv_data('dummy_path.csv', missing_strategy='mean')
    print("Data cleaning completed")