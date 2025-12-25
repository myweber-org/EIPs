
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
    
    return Trueimport pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop')
        columns (list): List of columns to process, None for all numeric columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=columns)
    elif strategy == 'mean':
        for col in columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    elif strategy == 'median':
        for col in columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif strategy == 'mode':
        for col in columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    return df_clean

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to process, None for all numeric columns
        threshold (float): IQR multiplier threshold
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    df_standardized = df.copy()
    
    if columns is None:
        columns = df_standardized.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        mean = df_standardized[col].mean()
        std = df_standardized[col].std()
        if std > 0:
            df_standardized[col] = (df_standardized[col] - mean) / std
    
    return df_standardized

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"