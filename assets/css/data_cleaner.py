
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values. 
                           Options: 'mean', 'median', 'drop', 'fill_zero'.
    outlier_threshold (float): Number of standard deviations to identify outliers.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    df_clean = df.copy()
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif missing_strategy == 'median':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    elif missing_strategy == 'drop':
        df_clean = df_clean.dropna(subset=numeric_cols)
    elif missing_strategy == 'fill_zero':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
    
    # Handle outliers using z-score method
    for col in numeric_cols:
        z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
        outlier_mask = z_scores > outlier_threshold
        
        if outlier_mask.any():
            # Cap outliers at threshold * standard deviation
            upper_bound = df_clean[col].mean() + outlier_threshold * df_clean[col].std()
            lower_bound = df_clean[col].mean() - outlier_threshold * df_clean[col].std()
            
            df_clean.loc[outlier_mask, col] = np.where(
                df_clean.loc[outlier_mask, col] > upper_bound,
                upper_bound,
                np.where(
                    df_clean.loc[outlier_mask, col] < lower_bound,
                    lower_bound,
                    df_clean.loc[outlier_mask, col]
                )
            )
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],  # Contains outlier (100) and missing value
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    df_cleaned = clean_dataframe(df, missing_strategy='mean', outlier_threshold=2)
    print("Cleaned DataFrame:")
    print(df_cleaned)
    print("\n")
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(df_cleaned, required_columns=['A', 'B', 'C'])
    print(f"Validation: {is_valid}")
    print(f"Message: {message}")import pandas as pd
import numpy as np
from typing import Optional, List

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values using specified strategy.
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_filled[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df_filled[col] = df[col].fillna(df[col].median())
            elif strategy == 'zero':
                df_filled[col] = df[col].fillna(0)
        else:
            df_filled[col] = df[col].fillna('Unknown')
    
    return df_filled

def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize numeric column to range [0, 1].
    """
    df_normalized = df.copy()
    if df[column].dtype in ['int64', 'float64']:
        col_min = df[column].min()
        col_max = df[column].max()
        
        if col_max != col_min:
            df_normalized[column] = (df[column] - col_min) / (col_max - col_min)
    
    return df_normalized

def remove_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from specified column.
    """
    if df[column].dtype not in ['int64', 'float64']:
        return df
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        z_scores = np.abs((df[column] - mean) / std)
        return df[z_scores <= threshold]
    
    return df

def clean_dataframe(df: pd.DataFrame, 
                    remove_dups: bool = True,
                    fill_na: bool = True,
                    fill_strategy: str = 'mean',
                    normalize_cols: Optional[List[str]] = None,
                    remove_outlier_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df, strategy=fill_strategy)
    
    if normalize_cols:
        for col in normalize_cols:
            cleaned_df = normalize_column(cleaned_df, col)
    
    if remove_outlier_cols:
        for col in remove_outlier_cols:
            cleaned_df = remove_outliers(cleaned_df, col)
    
    return cleaned_df