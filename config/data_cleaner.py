
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
    
    return filtered_df

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not clean column '{col}': {e}")
    
    return cleaned_df

def get_data_summary(df):
    """
    Generate summary statistics for a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    summary = {
        'original_rows': len(df),
        'cleaned_rows': None,
        'removed_rows': None,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns)
    }
    return summary

def process_dataset(file_path, output_path=None):
    """
    Load, clean, and optionally save a dataset.
    
    Parameters:
    file_path (str): Path to input CSV file
    output_path (str): Path to save cleaned CSV file (optional)
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {e}")
    
    original_summary = get_data_summary(df)
    print(f"Original dataset: {original_summary['original_rows']} rows")
    
    cleaned_df = clean_numeric_data(df)
    
    cleaned_summary = get_data_summary(cleaned_df)
    cleaned_summary['original_rows'] = original_summary['original_rows']
    cleaned_summary['removed_rows'] = original_summary['original_rows'] - cleaned_summary['original_rows']
    
    print(f"Cleaned dataset: {cleaned_summary['original_rows']} rows")
    print(f"Removed {cleaned_summary['removed_rows']} outlier rows")
    
    if output_path:
        cleaned_df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(2, 1000),
        'C': np.random.randint(1, 100, 1000),
        'category': np.random.choice(['X', 'Y', 'Z'], 1000)
    })
    
    sample_data.loc[10:15, 'A'] = 500
    sample_data.loc[20:25, 'B'] = 50
    
    print("Sample data created with artificial outliers")
    cleaned = clean_numeric_data(sample_data, ['A', 'B'])
    print(f"Original: {len(sample_data)} rows, Cleaned: {len(cleaned)} rows")