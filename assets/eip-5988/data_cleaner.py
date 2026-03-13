
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
    
    return filtered_df

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
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
        'count': df[column].count()
    }
    
    return stats

def process_numerical_data(df, columns=None):
    """
    Process multiple numerical columns by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, process all numerical columns.
    
    Returns:
    pd.DataFrame: Processed DataFrame with outliers removed
    dict: Dictionary of statistics for each processed column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    processed_df = df.copy()
    all_stats = {}
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            original_count = len(processed_df)
            processed_df = remove_outliers_iqr(processed_df, col)
            removed_count = original_count - len(processed_df)
            
            stats = calculate_statistics(processed_df, col)
            stats['outliers_removed'] = removed_count
            all_stats[col] = stats
    
    return processed_df, all_stats

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 100, 27, 28, 29, -10],
        'humidity': [45, 46, 47, 48, 49, 50, 200, 51, 52, -5],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 2000, 1020, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df, stats = process_numerical_data(df)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    print("Statistics:")
    for col, col_stats in stats.items():
        print(f"\n{col}:")
        for key, value in col_stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=np.nan):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
        print(f"Removed {len(df) - len(df_clean)} duplicate rows.")
    
    if fill_missing:
        missing_before = df_clean.isnull().sum().sum()
        df_clean = df_clean.fillna(fill_value)
        missing_after = df_clean.isnull().sum().sum()
        print(f"Filled {missing_before - missing_after} missing values.")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate the DataFrame structure and required columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def main():
    # Example usage
    data = {
        'A': [1, 2, 2, 3, None],
        'B': [4, None, 6, 6, 8],
        'C': [7, 8, 9, 9, 10]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaning data...")
    
    try:
        validate_dataframe(df, required_columns=['A', 'B', 'C'])
        df_clean = clean_dataframe(df, fill_value=0)
        print("\nCleaned DataFrame:")
        print(df_clean)
    except Exception as e:
        print(f"Error during data cleaning: {e}")

if __name__ == "__main__":
    main()