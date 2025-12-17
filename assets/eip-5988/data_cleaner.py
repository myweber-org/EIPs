
import pandas as pd
import numpy as np

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
    summary = df[numeric_cols].describe()
    
    summary.loc['variance'] = df[numeric_cols].var()
    summary.loc['skewness'] = df[numeric_cols].skew()
    summary.loc['kurtosis'] = df[numeric_cols].kurtosis()
    
    return summary

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    df_clean = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=numeric_cols)
    elif strategy == 'mean':
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    elif strategy == 'median':
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif strategy == 'mode':
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    else:
        raise ValueError("Invalid strategy. Choose from: 'mean', 'median', 'mode', 'drop'")
    
    return df_clean

def normalize_data(df, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            
            if max_val != min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    
    return df_normalized

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 60],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_outliers_iqr(df, 'A')
    print("DataFrame after outlier removal:")
    print(cleaned_df)
    print()
    
    stats = calculate_summary_statistics(df)
    print("Summary Statistics:")
    print(stats)
    print()
    
    normalized_df = normalize_data(df, ['A', 'B'])
    print("Normalized DataFrame:")
    print(normalized_df)