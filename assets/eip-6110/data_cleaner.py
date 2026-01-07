import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. 
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            If None, missing values are not filled.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or np.array): Input data
    column (int): Column index to process
    
    Returns:
    np.array: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    q1 = np.percentile(data[:, column], 25)
    q3 = np.percentile(data[:, column], 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (data[:, column] >= lower_bound) & (data[:, column] <= upper_bound)
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    data (np.array): Input data
    column (int): Column index
    
    Returns:
    dict: Dictionary containing statistics
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    col_data = data[:, column]
    
    stats = {
        'mean': np.mean(col_data),
        'median': np.median(col_data),
        'std': np.std(col_data),
        'min': np.min(col_data),
        'max': np.max(col_data),
        'count': len(col_data)
    }
    
    return stats

def normalize_data(data, column, method='minmax'):
    """
    Normalize data in a column.
    
    Parameters:
    data (np.array): Input data
    column (int): Column index
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    np.array: Normalized data
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    col_data = data[:, column].astype(float)
    
    if method == 'minmax':
        min_val = np.min(col_data)
        max_val = np.max(col_data)
        if max_val - min_val == 0:
            normalized = np.zeros_like(col_data)
        else:
            normalized = (col_data - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = np.mean(col_data)
        std_val = np.std(col_data)
        if std_val == 0:
            normalized = np.zeros_like(col_data)
        else:
            normalized = (col_data - mean_val) / std_val
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    data[:, column] = normalized
    return data

def process_dataset(data, column, remove_outliers=True, normalize=True):
    """
    Complete data processing pipeline.
    
    Parameters:
    data (list or np.array): Input data
    column (int): Column index to process
    remove_outliers (bool): Whether to remove outliers
    normalize (bool): Whether to normalize data
    
    Returns:
    tuple: (processed_data, original_stats, processed_stats)
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    original_stats = calculate_statistics(data, column)
    
    if remove_outliers:
        data = remove_outliers_iqr(data, column)
    
    if normalize:
        data = normalize_data(data, column)
    
    processed_stats = calculate_statistics(data, column)
    
    return data, original_stats, processed_stats