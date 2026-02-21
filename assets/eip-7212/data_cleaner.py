import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and optionally dropping columns.
    
    Parameters:
    filepath (str): Path to the CSV file.
    missing_strategy (str): Strategy for handling missing values.
                            Options: 'mean', 'median', 'drop', 'zero'.
    columns_to_drop (list): List of column names to drop from the dataset.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {filepath}")
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
        print(f"Dropped columns: {columns_to_drop}")
    
    if df.isnull().sum().any():
        print("Missing values detected:")
        print(df.isnull().sum())
        
        if missing_strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif missing_strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif missing_strategy == 'drop':
            df = df.dropna()
        elif missing_strategy == 'zero':
            df = df.fillna(0)
        else:
            raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    
    cleaned_shape = df.shape
    print(f"Cleaned data shape: {cleaned_shape}")
    print(f"Rows removed: {original_shape[0] - cleaned_shape[0]}")
    
    return df

def detect_outliers_iqr(df, column):
    """
    Detect outliers in a specific column using the IQR method.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    column (str): Column name to check for outliers.
    
    Returns:
    pandas.DataFrame: DataFrame containing only the outlier rows.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    print(f"Outlier detection for column '{column}':")
    print(f"  IQR: {IQR:.2f}")
    print(f"  Lower bound: {lower_bound:.2f}")
    print(f"  Upper bound: {upper_bound:.2f}")
    print(f"  Outliers found: {len(outliers)}")
    
    return outliers

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['x', 'y', 'z', 'x', 'y', 'z'],
        'D': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', missing_strategy='mean', columns_to_drop=['D'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    outliers = detect_outliers_iqr(cleaned_df, 'A')
    print("\nOutliers in column 'A':")
    print(outliers)