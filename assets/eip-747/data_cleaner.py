
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
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 200],
        'C': ['a', 'b', 'c', 'd', 'e', 'f']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nSummary:")
    print(get_data_summary(df))
    
    cleaned_df = clean_numeric_data(df, columns=['A', 'B'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
import pandas as pd
import numpy as np

def clean_dataset(df, key_columns=None, date_column=None, numeric_columns=None):
    """
    Clean a pandas DataFrame by removing duplicates, handling missing values,
    and standardizing column formats.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows based on key columns or all columns
    if key_columns:
        cleaned_df = cleaned_df.drop_duplicates(subset=key_columns, keep='first')
    else:
        cleaned_df = cleaned_df.drop_duplicates(keep='first')
    
    # Standardize date column format if specified
    if date_column and date_column in cleaned_df.columns:
        cleaned_df[date_column] = pd.to_datetime(cleaned_df[date_column], errors='coerce')
    
    # Fill missing numeric values with column median
    if numeric_columns:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                median_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(median_val)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content requirements.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def export_cleaned_data(df, output_path, format='csv'):
    """
    Export cleaned DataFrame to specified format.
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Data exported successfully to {output_path}")

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None],
        'date': ['2023-01-01', '2023-01-02', '2023-01-02', 'invalid', '2023-01-03'],
        'value': [100, 200, 200, None, 400]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned = clean_dataset(
        df, 
        key_columns=['id', 'name'],
        date_column='date',
        numeric_columns=['value']
    )
    
    print("Cleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    try:
        validate_data(cleaned, required_columns=['id', 'name', 'value'], min_rows=3)
        print("Data validation passed")
    except ValueError as e:
        print(f"Data validation failed: {e}")import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df: pd.DataFrame, 
                         column_types: dict) -> pd.DataFrame:
    """
    Convert specified columns to given data types.
    
    Args:
        df: Input DataFrame
        column_types: Dictionary mapping column names to target types
    
    Returns:
        DataFrame with converted column types
    """
    df_copy = df.copy()
    for column, dtype in column_types.items():
        if column in df_copy.columns:
            try:
                df_copy[column] = df_copy[column].astype(dtype)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert column '{column}' to {dtype}")
    return df_copy

def handle_missing_values(df: pd.DataFrame, 
                          strategy: str = 'drop',
                          fill_value: Optional[float] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill values
        fill_value: Value to use when filling (if strategy is 'fill')
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill' and fill_value is not None:
        return df.fillna(fill_value)
    else:
        return df

def normalize_numeric_columns(df: pd.DataFrame,
                              columns: List[str],
                              method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize numeric columns using specified method.
    
    Args:
        df: Input DataFrame
        columns: List of column names to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized columns
    """
    df_copy = df.copy()
    
    for column in columns:
        if column in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[column]):
            if method == 'minmax':
                col_min = df_copy[column].min()
                col_max = df_copy[column].max()
                if col_max != col_min:
                    df_copy[column] = (df_copy[column] - col_min) / (col_max - col_min)
            elif method == 'zscore':
                col_mean = df_copy[column].mean()
                col_std = df_copy[column].std()
                if col_std != 0:
                    df_copy[column] = (df_copy[column] - col_mean) / col_std
    
    return df_copy

def clean_dataframe(df: pd.DataFrame,
                    remove_dups: bool = True,
                    type_conversions: Optional[dict] = None,
                    missing_strategy: str = 'drop',
                    normalize_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        remove_dups: Whether to remove duplicates
        type_conversions: Dictionary of column type conversions
        missing_strategy: Strategy for handling missing values
        normalize_cols: Columns to normalize
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize_cols:
        cleaned_df = normalize_numeric_columns(cleaned_df, normalize_cols)
    
    return cleaned_df