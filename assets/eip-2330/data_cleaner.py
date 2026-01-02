
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    numeric_check = df[required_columns].select_dtypes(include=[np.number])
    if len(numeric_check.columns) != len(required_columns):
        non_numeric = [col for col in required_columns if col not in numeric_check.columns]
        raise ValueError(f"Non-numeric columns found: {non_numeric}")
    
    return True
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, handle_nulls='drop', fill_value=0):
    """
    Clean a pandas DataFrame by handling duplicates and null values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to drop duplicate rows
    handle_nulls (str): Method to handle nulls - 'drop', 'fill', or 'ignore'
    fill_value: Value to fill nulls with if handle_nulls='fill'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if handle_nulls == 'drop':
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} rows with null values")
    elif handle_nulls == 'fill':
        cleaned_df = cleaned_df.fillna(fill_value)
        print(f"Filled null values with {fill_value}")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: Less than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing columns {missing_cols}")
            return False
    
    return True

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to check for outliers
    method (str): 'iqr' for interquartile range or 'zscore' for standard deviation
    threshold (float): Threshold multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        print(f"Column {column} not found in DataFrame")
        return df
    
    data = df[column].dropna()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        mask = (df[column] - mean).abs() <= threshold * std
    else:
        print(f"Unknown method: {method}")
        return df
    
    outliers_removed = len(df) - mask.sum()
    print(f"Removed {outliers_removed} outliers from column '{column}'")
    
    return df[mask]

def normalize_column(df, column, method='minmax'):
    """
    Normalize values in a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): 'minmax' or 'zscore' normalization
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        print(f"Column {column} not found in DataFrame")
        return df
    
    normalized_df = df.copy()
    
    if method == 'minmax':
        col_min = normalized_df[column].min()
        col_max = normalized_df[column].max()
        if col_max != col_min:
            normalized_df[column] = (normalized_df[column] - col_min) / (col_max - col_min)
        else:
            normalized_df[column] = 0
    elif method == 'zscore':
        col_mean = normalized_df[column].mean()
        col_std = normalized_df[column].std()
        if col_std > 0:
            normalized_df[column] = (normalized_df[column] - col_mean) / col_std
        else:
            normalized_df[column] = 0
    
    return normalized_df

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10, 20, None, 40, 50, 50, 1000],
        'category': ['A', 'B', 'C', 'A', 'B', 'B', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, handle_nulls='fill', fill_value=0)
    print("After cleaning:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['id', 'value', 'category'])
    print(f"\nData validation: {is_valid}")
    
    no_outliers = remove_outliers(cleaned, 'value', method='iqr')
    print(f"\nAfter outlier removal: {len(no_outliers)} rows")
    
    normalized = normalize_column(no_outliers, 'value', method='minmax')
    print("\nAfter normalization:")
    print(normalized)