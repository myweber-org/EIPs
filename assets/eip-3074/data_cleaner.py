
import pandas as pd
import numpy as np

def clean_csv_data(file_path, strategy='mean', output_path=None):
    """
    Clean missing values in a CSV file using specified strategy.
    
    Args:
        file_path (str): Path to input CSV file
        strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        output_path (str, optional): Path for cleaned output file
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    
    df = pd.read_csv(file_path)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
    elif strategy == 'drop':
        df.dropna(subset=numeric_cols, inplace=True)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        threshold (float): IQR multiplier threshold
    
    Returns:
        pd.Series: Boolean mask of outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return outliers

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to normalize
        method (str): Normalization method ('minmax', 'zscore')
    
    Returns:
        pd.DataFrame: DataFrame with normalized column
    """
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return df_copy

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': ['x', 'y', 'z', 'x', 'y']
    })
    
    cleaned_df = clean_csv_data('sample.csv', strategy='mean')
    outliers = detect_outliers_iqr(cleaned_df, 'A')
    normalized_df = normalize_column(cleaned_df, 'B', method='minmax')
    
    print("Data cleaning completed successfully")