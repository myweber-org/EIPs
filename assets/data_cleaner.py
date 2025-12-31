
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
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
    Example demonstrating the usage of data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': range(100),
        'value': np.concatenate([
            np.random.normal(100, 10, 90),
            np.array([500, -200, 300, -150, 400])
        ])
    }
    
    df = pd.DataFrame(data)
    
    print("Original data shape:", df.shape)
    print("Original statistics:", calculate_summary_statistics(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_df, 'value'))
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_data.head())
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    result = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_numeric_data(values, default=0):
    """
    Clean numeric data by converting strings to floats,
    handling None values, and replacing invalid entries with default.
    """
    cleaned = []
    for val in values:
        if val is None:
            cleaned.append(default)
        elif isinstance(val, str):
            try:
                cleaned.append(float(val))
            except ValueError:
                cleaned.append(default)
        else:
            cleaned.append(float(val))
    return cleaned

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    print("Original:", sample_data)
    print("Cleaned:", remove_duplicates(sample_data))
    
    numeric_data = ["1.5", "invalid", None, 3.2, "4.7"]
    print("Numeric data:", numeric_data)
    print("Cleaned numeric:", clean_numeric_data(numeric_data))