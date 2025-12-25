
import pandas as pd
import numpy as np
from typing import Optional, Union, List

def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    factor: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers from specified columns using IQR method.
    
    Parameters:
    df: Input DataFrame
    columns: Column name or list of column names to process
    factor: IQR multiplier for outlier detection
    
    Returns:
    DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    elif isinstance(columns, str):
        columns = [columns]
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mean',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Parameters:
    df: Input DataFrame
    strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns: Specific columns to process
    
    Returns:
    DataFrame with handled missing values
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'drop':
            df_filled = df_filled.dropna(subset=[col])
        elif strategy in ['mean', 'median']:
            if pd.api.types.is_numeric_dtype(df[col]):
                if strategy == 'mean':
                    fill_value = df[col].mean()
                else:
                    fill_value = df[col].median()
                df_filled[col] = df_filled[col].fillna(fill_value)
        elif strategy == 'mode':
            if not df[col].empty:
                fill_value = df[col].mode()[0] if not df[col].mode().empty else None
                df_filled[col] = df_filled[col].fillna(fill_value)
    
    return df_filled.reset_index(drop=True)

def normalize_data(
    df: pd.DataFrame,
    method: str = 'minmax',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Normalize numerical columns in DataFrame.
    
    Parameters:
    df: Input DataFrame
    method: Normalization method ('minmax', 'zscore')
    columns: Specific columns to normalize
    
    Returns:
    DataFrame with normalized columns
    """
    df_norm = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df_norm[col] = (df[col] - mean_val) / std_val
    
    return df_norm

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1
) -> bool:
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df: DataFrame to validate
    required_columns: List of columns that must be present
    min_rows: Minimum number of rows required
    
    Returns:
    Boolean indicating if DataFrame is valid
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if len(df) < min_rows:
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False
    
    return True