
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(input_file, output_file, outlier_method='iqr', normalize_method='minmax'):
    df = load_data(input_file)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            df = remove_outliers_iqr(df, col)
        elif outlier_method == 'zscore':
            df = remove_outliers_zscore(df, col)
    
    for col in numeric_cols:
        if normalize_method == 'minmax':
            df = normalize_minmax(df, col)
        elif normalize_method == 'zscore':
            df = normalize_zscore(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')import pandas as pd
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

def normalize_column(df: pd.DataFrame, column: str, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize specified column using given method.
    
    Args:
        df: Input DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    df = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            df[column] = (df[column] - mean_val) / std_val
    
    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in numeric columns.
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', or 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df.dropna(subset=numeric_cols)
    
    for col in numeric_cols:
        if strategy == 'mean':
            fill_value = df[col].mean()
        elif strategy == 'median':
            fill_value = df[col].median()
        else:
            continue
        
        df[col] = df[col].fillna(fill_value)
    
    return df

def clean_dataset(df: pd.DataFrame, 
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
    
    return cleaned_dfimport pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options are 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (bool, str) indicating validation result and message.
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (mean imputation):")
    cleaned = clean_dataset(df, fill_missing='mean')
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned)
    print(f"\nValidation: {is_valid}, Message: {message}")