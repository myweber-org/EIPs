
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
    
    return filtered_df.reset_index(drop=True)

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistics
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df[column])
    }
    return stats

def example_usage():
    """
    Example demonstrating how to use the data cleaning functions.
    """
    np.random.seed(42)
    data = pd.DataFrame({
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.array([300, 350, -50, -100])
        ])
    })
    
    print("Original data statistics:")
    print(calculate_statistics(data, 'values'))
    
    cleaned_data = remove_outliers_iqr(data, 'values')
    
    print("\nCleaned data statistics:")
    print(calculate_statistics(cleaned_data, 'values'))
    
    print(f"\nRemoved {len(data) - len(cleaned_data)} outliers")
    
    return cleaned_data

if __name__ == "__main__":
    cleaned = example_usage()