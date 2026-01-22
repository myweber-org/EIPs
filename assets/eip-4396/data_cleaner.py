import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_na=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_na (str or dict): Method to fill missing values.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if fill_na is not None:
        cleaned_df = cleaned_df.fillna(fill_na)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Data validation passed"
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    factor (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def z_score_normalize(data, column):
    """
    Normalize data using Z-score method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Z-score normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    z_scores = (data[column] - mean_val) / std_val
    return z_scores

def detect_skewed_columns(data, threshold=0.5):
    """
    Detect columns with skewed distributions.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    threshold (float): Absolute skewness threshold (default 0.5)
    
    Returns:
    list: Column names with absolute skewness > threshold
    """
    skewed_cols = []
    
    for col in data.select_dtypes(include=[np.number]).columns:
        skewness = data[col].skew()
        if abs(skewness) > threshold:
            skewed_cols.append((col, skewness))
    
    return skewed_cols

def log_transform(data, column):
    """
    Apply log transformation to reduce skewness.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to transform
    
    Returns:
    pd.Series: Log-transformed values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    if data[column].min() <= 0:
        shifted = data[column] - data[column].min() + 1
        transformed = np.log(shifted)
    else:
        transformed = np.log(data[column])
    
    return transformed

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5, normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric columns to process
    outlier_factor (float): IQR factor for outlier removal
    normalize_method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if col not in cleaned_data.columns:
            continue
            
        original_len = len(cleaned_data)
        cleaned_data = remove_outliers_iqr(cleaned_data, col, outlier_factor)
        removed_count = original_len - len(cleaned_data)
        
        if removed_count > 0:
            print(f"Removed {removed_count} outliers from column '{col}'")
        
        if normalize_method == 'minmax':
            cleaned_data[f'{col}_normalized'] = normalize_minmax(cleaned_data, col)
        elif normalize_method == 'zscore':
            cleaned_data[f'{col}_normalized'] = z_score_normalize(cleaned_data, col)
        else:
            raise ValueError("normalize_method must be 'minmax' or 'zscore'")
    
    skewed = detect_skewed_columns(cleaned_data[numeric_columns])
    for col, skew_val in skewed:
        if abs(skew_val) > 1.0:
            cleaned_data[f'{col}_log'] = log_transform(cleaned_data, col)
            print(f"Applied log transform to '{col}' (skewness: {skew_val:.2f})")
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan_ratio=0.1):
    """
    Validate data quality and completeness.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    required_columns (list): List of required columns
    allow_nan_ratio (float): Maximum allowed NaN ratio per column
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'missing_columns': [],
        'high_nan_columns': []
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing}")
    
    for col in data.columns:
        nan_ratio = data[col].isna().mean()
        if nan_ratio > allow_nan_ratio:
            validation_results['high_nan_columns'].append((col, nan_ratio))
            validation_results['is_valid'] = False
            validation_results['issues'].append(
                f"Column '{col}' has {nan_ratio:.1%} missing values"
            )
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].std() == 0:
            validation_results['issues'].append(
                f"Column '{col}' has zero variance"
            )
    
    return validation_results