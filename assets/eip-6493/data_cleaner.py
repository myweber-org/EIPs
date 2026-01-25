
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def z_score_normalization(data, column):
    """
    Apply z-score normalization to a column.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data = data.copy()
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        data[f'{column}_normalized'] = 0
    else:
        data[f'{column}_normalized'] = (data[column] - mean_val) / std_val
    
    return data

def min_max_normalization(data, column, feature_range=(0, 1)):
    """
    Apply min-max normalization to a column.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
        feature_range: desired range of transformed data (default 0-1)
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data = data.copy()
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        data[f'{column}_scaled'] = feature_range[0]
    else:
        data[f'{column}_scaled'] = (data[column] - min_val) / (max_val - min_val)
        data[f'{column}_scaled'] = data[f'{column}_scaled'] * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    return data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of columns to process (None processes all numeric columns)
    
    Returns:
        DataFrame with missing values handled
    """
    data = data.copy()
    
    if columns is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for column in columns:
        if column not in data.columns:
            continue
            
        if strategy == 'drop':
            data = data.dropna(subset=[column])
        elif strategy == 'mean':
            data[column] = data[column].fillna(data[column].mean())
        elif strategy == 'median':
            data[column] = data[column].fillna(data[column].median())
        elif strategy == 'mode':
            mode_val = data[column].mode()
            if not mode_val.empty:
                data[column] = data[column].fillna(mode_val.iloc[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return data

def clean_dataset(data, config):
    """
    Apply multiple cleaning operations based on configuration.
    
    Args:
        data: pandas DataFrame
        config: dictionary with cleaning configuration
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if 'missing_values' in config:
        strategy = config['missing_values'].get('strategy', 'mean')
        columns = config['missing_values'].get('columns')
        cleaned_data = handle_missing_values(cleaned_data, strategy, columns)
    
    if 'outliers' in config:
        for column_config in config['outliers']:
            column = column_config['column']
            method = column_config.get('method', 'iqr')
            multiplier = column_config.get('multiplier', 1.5)
            
            if method == 'iqr':
                cleaned_data = remove_outliers_iqr(cleaned_data, column, multiplier)
    
    if 'normalization' in config:
        for norm_config in config['normalization']:
            column = norm_config['column']
            method = norm_config.get('method', 'zscore')
            
            if method == 'zscore':
                cleaned_data = z_score_normalization(cleaned_data, column)
            elif method == 'minmax':
                feature_range = norm_config.get('feature_range', (0, 1))
                cleaned_data = min_max_normalization(cleaned_data, column, feature_range)
    
    return cleaned_data
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df[column]),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original summary statistics:")
    print(calculate_summary_stats(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary statistics:")
    print(calculate_summary_stats(cleaned_df, 'value'))
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()