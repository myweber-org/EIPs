import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize a numeric column to range [0, 1].
    
    Args:
        df: Input DataFrame
        column: Name of column to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    df = df.copy()
    col_min = df[column].min()
    col_max = df[column].max()
    
    if col_max == col_min:
        df[f'{column}_normalized'] = 0.5
    else:
        df[f'{column}_normalized'] = (df[column] - col_min) / (col_max - col_min)
    
    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in numeric columns.
    
    Args:
        df: Input DataFrame
        strategy: Method for imputation ('mean', 'median', 'zero')
    
    Returns:
        DataFrame with missing values handled
    """
    valid_strategies = ['mean', 'median', 'zero']
    if strategy not in valid_strategies:
        raise ValueError(f"Strategy must be one of {valid_strategies}")
    
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            else:
                fill_value = 0
            
            df[col] = df[col].fillna(fill_value)
    
    return df

def clean_dataframe(df: pd.DataFrame, 
                   deduplicate: bool = True,
                   normalize_cols: Optional[List[str]] = None,
                   missing_strategy: str = 'mean') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        normalize_cols: Columns to normalize
        missing_strategy: Strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col)
    
    return cleaned_df
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (bool): Whether to fill missing values
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'zero')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        for column in df_clean.columns:
            if df_clean[column].isnull().any():
                if df_clean[column].dtype in ['int64', 'float64']:
                    if fill_strategy == 'mean':
                        fill_value = df_clean[column].mean()
                    elif fill_strategy == 'median':
                        fill_value = df_clean[column].median()
                    elif fill_strategy == 'mode':
                        fill_value = df_clean[column].mode()[0]
                    elif fill_strategy == 'zero':
                        fill_value = 0
                    else:
                        fill_value = df_clean[column].mean()
                    
                    df_clean[column] = df_clean[column].fillna(fill_value)
                    print(f"Filled missing values in '{column}' with {fill_strategy}: {fill_value}")
                else:
                    df_clean[column] = df_clean[column].fillna('Unknown')
                    print(f"Filled missing values in '{column}' with 'Unknown'")
    
    return df_clean

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['all_required_columns_present'] = len(missing_columns) == 0
    
    return validation_results

def sample_usage():
    """Demonstrate usage of the data cleaning functions."""
    np.random.seed(42)
    
    data = {
        'id': [1, 2, 2, 3, 4, 5, 5, 6],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0, 50.0, np.nan],
        'category': ['A', 'B', 'B', 'C', np.nan, 'A', 'A', 'B'],
        'score': [85, 92, 92, 78, 88, np.nan, np.nan, 95]
    }
    
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df)
    print("\nDataset info:")
    print(df.info())
    
    cleaned_df = clean_dataset(df, fill_strategy='median')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    validation = validate_dataset(cleaned_df, required_columns=['id', 'value', 'category'])
    print("\nValidation results:")
    for key, value in validation.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    sample_usage()