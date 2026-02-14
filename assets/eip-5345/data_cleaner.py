
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_column(dataframe, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to normalize
        method: 'minmax' or 'zscore' normalization
    
    Returns:
        DataFrame with normalized column added as '{column}_normalized'
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result_df = dataframe.copy()
    
    if method == 'minmax':
        min_val = result_df[column].min()
        max_val = result_df[column].max()
        
        if max_val == min_val:
            result_df[f'{column}_normalized'] = 0.5
        else:
            result_df[f'{column}_normalized'] = (result_df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = result_df[column].mean()
        std_val = result_df[column].std()
        
        if std_val == 0:
            result_df[f'{column}_normalized'] = 0
        else:
            result_df[f'{column}_normalized'] = (result_df[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return result_df

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5, normalize=True):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: Input DataFrame
        numeric_columns: List of numeric columns to process (default: all numeric)
        outlier_threshold: IQR threshold for outlier removal
        normalize: Whether to normalize columns
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            # Remove outliers
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
            
            # Normalize if requested
            if normalize:
                cleaned_df = normalize_column(cleaned_df, col, method='minmax')
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(dataframe, required_columns=None, allow_nan=False):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: DataFrame to validate
        required_columns: List of required column names
        allow_nan: Whether NaN values are allowed
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if dataframe.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if not allow_nan and dataframe.isnull().any().any():
        nan_cols = dataframe.columns[dataframe.isnull().any()].tolist()
        return False, f"NaN values found in columns: {nan_cols}"
    
    return True, "DataFrame is valid"