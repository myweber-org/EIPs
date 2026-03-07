
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
    
    return filtered_df

def zscore_normalize(dataframe, column):
    """
    Normalize a column using z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to normalize
    
    Returns:
        DataFrame with normalized column added as '{column}_normalized'
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    normalized_col = f"{column}_normalized"
    dataframe[normalized_col] = stats.zscore(dataframe[column])
    
    return dataframe

def minmax_normalize(dataframe, column, feature_range=(0, 1)):
    """
    Normalize a column using min-max scaling.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to normalize
        feature_range: Desired range of transformed data (default 0-1)
    
    Returns:
        DataFrame with normalized column added as '{column}_scaled'
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    if max_val == min_val:
        raise ValueError(f"Column '{column}' has constant values")
    
    scaled_col = f"{column}_scaled"
    dataframe[scaled_col] = (dataframe[column] - min_val) / (max_val - min_val)
    
    if feature_range != (0, 1):
        min_target, max_target = feature_range
        dataframe[scaled_col] = dataframe[scaled_col] * (max_target - min_target) + min_target
    
    return dataframe

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame to clean
        numeric_columns: List of numeric columns to process (default: all numeric)
        outlier_threshold: IQR threshold for outlier removal
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column, outlier_threshold)
            removed_count = original_count - len(cleaned_df)
            
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column '{column}'")
            
            cleaned_df = zscore_normalize(cleaned_df, column)
            cleaned_df = minmax_normalize(cleaned_df, column)
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: List of columns that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if dataframe.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            elif fill_missing == 'mode':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary containing validation results.
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'empty_rows': 0,
        'total_rows': len(df)
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    empty_rows = df.isnull().all(axis=1).sum()
    validation_results['empty_rows'] = int(empty_rows)
    
    if empty_rows > 0:
        validation_results['is_valid'] = False
    
    return validation_results