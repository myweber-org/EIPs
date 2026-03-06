import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Strategy to fill missing values. 
                        Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            elif fill_missing == 'mode':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'x', 'z', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (drop duplicates, fill with mean):")
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned)
    print(f"\nValidation: {message}")
import pandas as pd
import numpy as np
from typing import List, Optional

def clean_dataset(df: pd.DataFrame, 
                  drop_duplicates: bool = True,
                  columns_to_standardize: Optional[List[str]] = None,
                  date_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Clean a pandas DataFrame by handling duplicates, standardizing text,
    and parsing dates.
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
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
                cleaned_df[col] = cleaned_df[col].replace({'nan': np.nan, 'none': np.nan})
    
    if date_columns:
        for col in date_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
    
    missing_values = cleaned_df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Dataset contains {missing_values} missing values")
    
    return cleaned_df

def validate_email_column(df: pd.DataFrame, email_column: str) -> pd.Series:
    """
    Validate email addresses in a specified column.
    Returns a boolean Series indicating valid emails.
    """
    import re
    
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    def is_valid_email(email):
        if pd.isna(email):
            return False
        return bool(re.match(email_pattern, str(email).strip()))
    
    return df[email_column].apply(is_valid_email)

def remove_outliers_iqr(df: pd.DataFrame, 
                       numeric_columns: List[str],
                       multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using the Interquartile Range method.
    """
    df_clean = df.copy()
    
    for col in numeric_columns:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            initial_count = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & 
                               (df_clean[col] <= upper_bound)]
            removed = initial_count - len(df_clean)
            print(f"Removed {removed} outliers from column '{col}'")
    
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', 'Alice Brown'],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.com', 'alice@domain.net'],
        'age': [25, 30, 25, 150, 28],
        'join_date': ['2023-01-15', '2023-02-20', '2023-01-15', '2023-03-10', '2023-04-05']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(
        df,
        drop_duplicates=True,
        columns_to_standardize=['name'],
        date_columns=['join_date']
    )
    
    print("Cleaned dataset:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    valid_emails = validate_email_column(cleaned, 'email')
    print("Email validation results:")
    print(valid_emails)
    print("\n" + "="*50 + "\n")
    
    no_outliers = remove_outliers_iqr(cleaned, ['age'])
    print("Dataset after outlier removal:")
    print(no_outliers)