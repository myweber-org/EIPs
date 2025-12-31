import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val - min_val == 0:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    if std_val == 0:
        return df[column]
    return (df[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_threshold=1.5):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    return cleaned_df

def process_features(df, feature_columns, method='standardize'):
    processed_df = df.copy()
    for col in feature_columns:
        if col in processed_df.columns:
            if method == 'normalize':
                processed_df[col] = normalize_minmax(processed_df, col)
            elif method == 'standardize':
                processed_df[col] = standardize_zscore(processed_df, col)
    return processed_df
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

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

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to standardize
    
    Returns:
    pd.Series: Standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to process, None for all numeric columns
    
    Returns:
    pd.DataFrame: Dataframe with missing values handled
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    result = data.copy()
    
    for col in columns:
        if col not in result.columns:
            continue
            
        if strategy == 'drop':
            result = result.dropna(subset=[col])
        elif strategy == 'mean':
            result[col] = result[col].fillna(result[col].mean())
        elif strategy == 'median':
            result[col] = result[col].fillna(result[col].median())
        elif strategy == 'mode':
            result[col] = result[col].fillna(result[col].mode()[0] if not result[col].mode().empty else 0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return result

def create_cleaning_pipeline(data, config):
    """
    Apply multiple cleaning operations based on configuration.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    config (dict): Configuration dictionary with cleaning operations
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_data = data.copy()
    
    if 'remove_outliers' in config:
        for col_config in config['remove_outliers']:
            col = col_config.get('column')
            multiplier = col_config.get('multiplier', 1.5)
            cleaned_data = remove_outliers_iqr(cleaned_data, col, multiplier)
    
    if 'normalize' in config:
        for col in config['normalize']:
            if col in cleaned_data.columns:
                cleaned_data[f'{col}_normalized'] = normalize_minmax(cleaned_data, col)
    
    if 'standardize' in config:
        for col in config['standardize']:
            if col in cleaned_data.columns:
                cleaned_data[f'{col}_standardized'] = standardize_zscore(cleaned_data, col)
    
    if 'handle_missing' in config:
        missing_config = config['handle_missing']
        strategy = missing_config.get('strategy', 'mean')
        columns = missing_config.get('columns')
        cleaned_data = handle_missing_values(cleaned_data, strategy, columns)
    
    return cleaned_data

def validate_data(data, checks):
    """
    Validate data against specified checks.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    checks (dict): Dictionary of validation checks
    
    Returns:
    dict: Dictionary with validation results
    """
    results = {}
    
    if 'non_negative' in checks:
        for col in checks['non_negative']:
            if col in data.columns:
                invalid_count = (data[col] < 0).sum()
                results[f'{col}_non_negative'] = {
                    'valid': invalid_count == 0,
                    'invalid_count': int(invalid_count)
                }
    
    if 'range_check' in checks:
        for check in checks['range_check']:
            col = check.get('column')
            min_val = check.get('min')
            max_val = check.get('max')
            
            if col in data.columns:
                below_min = (data[col] < min_val).sum() if min_val is not None else 0
                above_max = (data[col] > max_val).sum() if max_val is not None else 0
                
                results[f'{col}_range_check'] = {
                    'valid': below_min == 0 and above_max == 0,
                    'below_min': int(below_min),
                    'above_max': int(above_max)
                }
    
    return results