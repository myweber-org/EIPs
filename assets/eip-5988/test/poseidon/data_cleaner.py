
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Columns to consider for duplicates
    keep (str): Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    if subset is None:
        subset = df.columns.tolist()
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df.reset_index(drop=True)

def clean_numeric_columns(df, columns=None):
    """
    Clean numeric columns by replacing invalid values with NaN.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list, optional): Specific columns to clean
    
    Returns:
    pd.DataFrame: DataFrame with cleaned numeric columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list, optional): Columns that must be present
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'empty_rows': 0,
        'null_values': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    validation_results['empty_rows'] = df.isnull().all(axis=1).sum()
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            validation_results['null_values'][col] = null_count
    
    return validation_results
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    """
    Remove outliers from specified column using IQR method.
    
    Args:
        dataframe: pandas DataFrame containing the data
        column: column name to process
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def calculate_basic_stats(dataframe, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        dataframe: pandas DataFrame
        column: column name
    
    Returns:
        Dictionary containing statistics
    """
    stats = {
        'mean': dataframe[column].mean(),
        'median': dataframe[column].median(),
        'std': dataframe[column].std(),
        'min': dataframe[column].min(),
        'max': dataframe[column].max(),
        'count': len(dataframe)
    }
    return stats

def normalize_column(dataframe, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to normalize
        method: normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column added as '{column}_normalized'
    """
    if method == 'minmax':
        min_val = dataframe[column].min()
        max_val = dataframe[column].max()
        normalized = (dataframe[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = dataframe[column].mean()
        std_val = dataframe[column].std()
        normalized = (dataframe[column] - mean_val) / std_val
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    new_column_name = f"{column}_normalized"
    dataframe[new_column_name] = normalized
    return dataframe

def example_usage():
    """Example usage of the data cleaning functions."""
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 50, 100)
    }
    
    df = pd.DataFrame(data)
    print(f"Original data shape: {df.shape}")
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    print(f"Cleaned data shape: {cleaned_df.shape}")
    
    stats = calculate_basic_stats(cleaned_df, 'value')
    print(f"Statistics: {stats}")
    
    normalized_df = normalize_column(cleaned_df, 'value', method='zscore')
    print(f"Normalized column added: 'value_normalized'")
    
    return normalized_df

if __name__ == "__main__":
    result_df = example_usage()
    print(f"Final DataFrame columns: {result_df.columns.tolist()}")