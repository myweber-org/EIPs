
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
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
    
    return filtered_df

def calculate_summary_statistics(df):
    """
    Calculate summary statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return pd.DataFrame()
    
    stats = df[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max'])
    stats.loc['count'] = df[numeric_cols].count()
    
    return stats.T

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.Series: Normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'minmax':
        col_min = df[column].min()
        col_max = df[column].max()
        if col_max == col_min:
            return pd.Series([0] * len(df), index=df.index)
        return (df[column] - col_min) / (col_max - col_min)
    
    elif method == 'zscore':
        col_mean = df[column].mean()
        col_std = df[column].std()
        if col_std == 0:
            return pd.Series([0] * len(df), index=df.index)
        return (df[column] - col_mean) / col_std
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df.dropna(subset=numeric_cols)
    
    df_filled = df.copy()
    
    for col in numeric_cols:
        if strategy == 'mean':
            fill_value = df[col].mean()
        elif strategy == 'median':
            fill_value = df[col].median()
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
        
        df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Introduce some outliers
    df.loc[10, 'value'] = 300
    df.loc[20, 'value'] = -50
    
    print("Original DataFrame shape:", df.shape)
    print("\nSummary statistics before cleaning:")
    print(calculate_summary_statistics(df))
    
    # Clean data
    cleaned_df = remove_outliers_iqr(df, 'value')
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    
    # Handle missing values (none in this example)
    filled_df = handle_missing_values(cleaned_df, strategy='mean')
    
    # Normalize
    filled_df['value_normalized'] = normalize_column(filled_df, 'value', method='minmax')
    
    print("\nFinal summary statistics:")
    print(calculate_summary_statistics(filled_df))