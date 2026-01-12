
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    threshold (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
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

def zscore_normalize(dataframe, columns=None):
    """
    Apply z-score normalization to specified columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if normalized_df[col].dtype in [np.float64, np.int64]:
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            
            if std_val > 0:
                normalized_df[col] = (normalized_df[col] - mean_val) / std_val
            else:
                normalized_df[col] = 0
    
    return normalized_df

def minmax_normalize(dataframe, columns=None, feature_range=(0, 1)):
    """
    Apply min-max normalization to specified columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize
    feature_range (tuple): Desired range of transformed data
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    normalized_df = dataframe.copy()
    min_range, max_range = feature_range
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if normalized_df[col].dtype in [np.float64, np.int64]:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            
            if max_val > min_val:
                normalized_df[col] = ((normalized_df[col] - min_val) / 
                                     (max_val - min_val)) * (max_range - min_range) + min_range
            else:
                normalized_df[col] = min_range
    
    return normalized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'constant')
    columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: DataFrame with missing values handled
    """
    if columns is None:
        columns = dataframe.columns
    
    processed_df = dataframe.copy()
    
    for col in columns:
        if col not in processed_df.columns:
            continue
        
        if processed_df[col].isnull().any():
            if strategy == 'mean' and processed_df[col].dtype in [np.float64, np.int64]:
                fill_value = processed_df[col].mean()
            elif strategy == 'median' and processed_df[col].dtype in [np.float64, np.int64]:
                fill_value = processed_df[col].median()
            elif strategy == 'mode':
                fill_value = processed_df[col].mode()[0] if not processed_df[col].mode().empty else 0
            elif strategy == 'constant':
                fill_value = 0
            else:
                fill_value = processed_df[col].mean()
            
            processed_df[col] = processed_df[col].fillna(fill_value)
    
    return processed_df

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    dataframe (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame validation passed"