
import pandas as pd
import numpy as np
from typing import List, Optional

def clean_dataframe(df: pd.DataFrame, 
                    drop_duplicates: bool = True,
                    columns_to_standardize: Optional[List[str]] = None,
                    fill_missing: str = 'mean') -> pd.DataFrame:
    """
    Clean a pandas DataFrame by removing duplicates, standardizing specified columns,
    and handling missing values.
    
    Parameters:
    df: Input DataFrame
    drop_duplicates: Whether to remove duplicate rows
    columns_to_standardize: List of column names to standardize (lowercase, strip whitespace)
    fill_missing: Strategy for filling missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if columns_to_standardize:
        for col in columns_to_standardize:
            if col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].astype(str).str.lower().str.strip()
    
    if fill_missing != 'drop':
        for col in cleaned_df.columns:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    else:
        cleaned_df = cleaned_df.dropna()
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_email_column(df: pd.DataFrame, email_column: str) -> pd.Series:
    """
    Validate email addresses in a specified column.
    
    Parameters:
    df: Input DataFrame
    email_column: Name of the column containing email addresses
    
    Returns:
    Boolean Series indicating valid emails
    """
    import re
    
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    return df[email_column].astype(str).str.match(email_pattern)

def remove_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from a numeric column using the Interquartile Range method.
    
    Parameters:
    df: Input DataFrame
    column: Name of the column to clean
    multiplier: IQR multiplier for outlier detection
    
    Returns:
    DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} outliers from column '{column}'")
    
    return filtered_df

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', 'Alice Brown'],
        'email': ['john@example.com', 'jane@example.com', 'invalid-email', 'bob@example.com', 'alice@example.com'],
        'age': [25, 30, 25, 150, 28],
        'salary': [50000, 60000, 50000, 1000000, 55000]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataframe(df, 
                              columns_to_standardize=['name', 'email'],
                              fill_missing='mean')
    
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    email_valid = validate_email_column(cleaned, 'email')
    print("Valid emails:")
    print(email_valid)
    print("\n")
    
    no_outliers = remove_outliers_iqr(cleaned, 'salary')
    print("DataFrame without salary outliers:")
    print(no_outliers)