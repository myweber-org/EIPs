def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving the original order.
    Returns a new list with unique elements.
    """
    seen = set()
    unique_list = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, handle_nulls='drop'):
    """
    Clean a pandas DataFrame by removing duplicates and handling null values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        handle_nulls (str): How to handle null values - 'drop', 'fill_mean', 'fill_median', or 'fill_zero'
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if handle_nulls == 'drop':
        cleaned_df = cleaned_df.dropna()
        print(f"Dropped rows with null values")
    elif handle_nulls == 'fill_mean':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        print(f"Filled nulls with column means for numeric columns")
    elif handle_nulls == 'fill_median':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled nulls with column medians for numeric columns")
    elif handle_nulls == 'fill_zero':
        cleaned_df = cleaned_df.fillna(0)
        print(f"Filled all nulls with zeros")
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, message)
    """
    if len(df) < min_rows:
        return False, f"Dataset has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset validation passed"

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to check for outliers, None for all numeric columns
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    for col in columns:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    removed = initial_rows - len(df_clean)
    print(f"Removed {removed} rows containing outliers")
    
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5, 6],
        'value': [10.5, 20.3, 20.3, None, 15.7, 1000.0, 12.1],
        'category': ['A', 'B', 'B', 'C', None, 'A', 'D']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, handle_nulls='fill_mean')
    print("\nCleaned dataset:")
    print(cleaned)
    
    is_valid, message = validate_dataset(cleaned, required_columns=['id', 'value'])
    print(f"\nValidation: {message}")
    
    without_outliers = remove_outliers_iqr(cleaned, columns=['value'])
    print("\nDataset without outliers:")
    print(without_outliers)