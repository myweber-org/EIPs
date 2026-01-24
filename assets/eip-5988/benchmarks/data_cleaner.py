
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
    
    return filtered_df.copy()

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
        'count': df[column].count(),
        'q1': df[column].quantile(0.25),
        'q3': df[column].quantile(0.75)
    }
    
    return stats

def process_numerical_data(df, columns=None):
    """
    Process numerical columns by removing outliers and calculating statistics.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, process all numerical columns.
    
    Returns:
    tuple: (cleaned_df, statistics_dict)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    statistics = {}
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            try:
                original_count = len(cleaned_df)
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
                removed_count = original_count - len(cleaned_df)
                
                stats = calculate_summary_statistics(cleaned_df, col)
                stats['outliers_removed'] = removed_count
                statistics[col] = stats
                
            except Exception as e:
                print(f"Error processing column {col}: {str(e)}")
                continue
    
    return cleaned_df, statistics

def save_cleaned_data(df, output_path, statistics=None, stats_path=None):
    """
    Save cleaned data and optionally statistics to files.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame
    output_path (str): Path to save cleaned data
    statistics (dict): Statistics dictionary to save
    stats_path (str): Path to save statistics
    """
    df.to_csv(output_path, index=False)
    
    if statistics is not None and stats_path is not None:
        stats_df = pd.DataFrame(statistics).T
        stats_df.to_csv(stats_path)