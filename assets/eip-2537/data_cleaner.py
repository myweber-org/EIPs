import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, strategy='mean'):
    """
    Clean a CSV file by handling missing values.
    
    Parameters:
    input_path (str): Path to the input CSV file.
    output_path (str, optional): Path for the cleaned CSV file.
                                 If None, overwrites the input file.
    strategy (str): Method for handling missing values.
                    Options: 'mean', 'median', 'drop', 'zero'.
    
    Returns:
    pandas.DataFrame: The cleaned DataFrame.
    """
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    if strategy == 'drop':
        df_cleaned = df.dropna()
    elif strategy == 'mean':
        df_cleaned = df.fillna(df.select_dtypes(include=[np.number]).mean())
    elif strategy == 'median':
        df_cleaned = df.fillna(df.select_dtypes(include=[np.number]).median())
    elif strategy == 'zero':
        df_cleaned = df.fillna(0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    final_shape = df_cleaned.shape
    print(f"Cleaned data shape: {final_shape}")
    print(f"Rows removed: {original_shape[0] - final_shape[0]}")
    
    if output_path is None:
        output_path = input_path
    
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    
    return df_cleaned

def detect_outliers_iqr(df, column):
    """
    Detect outliers in a column using the IQR method.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    column (str): Column name to check for outliers.
    
    Returns:
    pandas.DataFrame: DataFrame containing outliers.
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
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
    print(f"  Found {len(outliers)} outliers out of {len(df)} rows")
    
    return outliers

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, np.nan, 30, 40, 50, 60],
        'C': ['x', 'y', 'z', np.nan, 'y', 'x']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_data.csv', strategy='mean')
    
    outliers = detect_outliers_iqr(cleaned_df, 'A')
    print("\nOutliers in column 'A':")
    print(outliers)