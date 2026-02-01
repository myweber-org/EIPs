
import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing values in a DataFrame using specified strategy.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for handling missing values. 
                       Options: 'mean', 'median', 'mode', 'drop', 'fill'
        columns (list): List of columns to apply cleaning to. If None, applies to all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=columns)
    elif strategy == 'mean':
        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    elif strategy == 'median':
        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif strategy == 'mode':
        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    elif strategy == 'fill':
        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)
    
    return df_clean

def detect_outliers(df, columns=None, threshold=3):
    """
    Detect outliers using z-score method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to check for outliers
        threshold (float): Z-score threshold for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outlier flags
    """
    df_outliers = df.copy()
    
    if columns is None:
        numeric_cols = df_outliers.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col in df_outliers.columns:
            z_scores = np.abs((df_outliers[col] - df_outliers[col].mean()) / df_outliers[col].std())
            df_outliers[f'{col}_outlier'] = z_scores > threshold
    
    return df_outliers

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize data using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to normalize
        method (str): Normalization method. Options: 'minmax', 'zscore'
    
    Returns:
        pd.DataFrame: Normalized DataFrame
    """
    df_norm = df.copy()
    
    if columns is None:
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    if method == 'minmax':
        for col in columns:
            if col in df_norm.columns:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val > min_val:
                    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        for col in columns:
            if col in df_norm.columns:
                mean_val = df_norm[col].mean()
                std_val = df_norm[col].std()
                if std_val > 0:
                    df_norm[col] = (df_norm[col] - mean_val) / std_val
    
    return df_norm

def validate_data(df, rules):
    """
    Validate data against specified rules.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        rules (dict): Dictionary of validation rules
    
    Returns:
        dict: Validation results
    """
    results = {}
    
    for col, rule in rules.items():
        if col in df.columns:
            if 'min' in rule:
                results[f'{col}_min_violations'] = (df[col] < rule['min']).sum()
            if 'max' in rule:
                results[f'{col}_max_violations'] = (df[col] > rule['max']).sum()
            if 'allowed_values' in rule:
                results[f'{col}_invalid_values'] = (~df[col].isin(rule['allowed_values'])).sum()
    
    return results

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, 20, 30, np.nan, 50],
        'C': [100, 200, 300, 400, 500]
    })
    
    print("Original Data:")
    print(sample_data)
    
    cleaned = clean_missing_data(sample_data, strategy='mean')
    print("\nCleaned Data:")
    print(cleaned)
    
    outliers = detect_outliers(cleaned)
    print("\nOutlier Detection:")
    print(outliers)
    
    normalized = normalize_data(cleaned, method='minmax')
    print("\nNormalized Data:")
    print(normalized)