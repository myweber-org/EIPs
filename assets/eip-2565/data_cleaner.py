
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    dataframe (pd.DataFrame): The input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def calculate_summary_statistics(dataframe, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    dataframe (pd.DataFrame): The input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': dataframe[column].mean(),
        'median': dataframe[column].median(),
        'std': dataframe[column].std(),
        'min': dataframe[column].min(),
        'max': dataframe[column].max(),
        'count': len(dataframe)
    }
    
    return stats

def clean_dataset(dataframe, numeric_columns=None):
    """
    Clean multiple numeric columns in a DataFrame by removing outliers.
    
    Parameters:
    dataframe (pd.DataFrame): The input DataFrame
    numeric_columns (list): List of column names to clean. If None, uses all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not clean column '{column}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original shape: {df.shape}")
    
    # Add some outliers
    df.loc[1000] = [500, 500, 500]
    df.loc[1001] = [-100, -100, -100]
    
    cleaned = clean_dataset(df)
    print(f"Cleaned shape: {cleaned.shape}")
    
    for col in ['A', 'B', 'C']:
        stats = calculate_summary_statistics(cleaned, col)
        print(f"\nStatistics for {col}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")