
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
    
    return filtered_df.reset_index(drop=True)

def detect_outliers_iqr(df, column):
    """
    Detect outliers in a DataFrame column using IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.Series: Boolean series indicating outliers (True = outlier)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return outliers

def calculate_statistics(df, column):
    """
    Calculate descriptive statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'q1': df[column].quantile(0.25),
        'q3': df[column].quantile(0.75),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.array([200, 250, -50, 300])
        ])
    })
    
    print("Original data shape:", sample_data.shape)
    print("Original statistics:", calculate_statistics(sample_data, 'values'))
    
    outliers = detect_outliers_iqr(sample_data, 'values')
    print(f"Number of outliers detected: {outliers.sum()}")
    
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print("Cleaned data shape:", cleaned_data.shape)
    print("Cleaned statistics:", calculate_statistics(cleaned_data, 'values'))