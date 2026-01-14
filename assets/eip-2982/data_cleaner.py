
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Columns to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean numeric column by handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Name of column to clean
        fill_method (str): Method to fill missing values ('mean', 'median', 'zero')
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if fill_method == 'mean':
        fill_value = df_copy[column_name].mean()
    elif fill_method == 'median':
        fill_value = df_copy[column_name].median()
    elif fill_method == 'zero':
        fill_value = 0
    else:
        raise ValueError("fill_method must be 'mean', 'median', or 'zero'")
    
    df_copy[column_name] = df_copy[column_name].fillna(fill_value)
    return df_copy

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import pandas as pd
import numpy as np

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

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    sample_data['A'][:5] = [500, 600, -300, 700, 800]
    
    df = pd.DataFrame(sample_data)
    print(f"Original dataset shape: {df.shape}")
    
    cleaned_df = clean_dataset(df)
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    
    print("\nSummary statistics:")
    print(cleaned_df.describe())