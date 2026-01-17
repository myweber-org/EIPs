import pandas as pd

def clean_dataset(df):
    """
    Remove null values and duplicate rows from a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def filter_by_column(df, column_name, min_value=None, max_value=None):
    """
    Filter DataFrame based on column value range.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Column to filter by.
        min_value: Minimum value for filtering (inclusive).
        max_value: Maximum value for filtering (inclusive).
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    filtered_df = df.copy()
    
    if min_value is not None:
        filtered_df = filtered_df[filtered_df[column_name] >= min_value]
    
    if max_value is not None:
        filtered_df = filtered_df[filtered_df[column_name] <= max_value]
    
    return filtered_dfimport numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    factor (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling to range [0, 1].
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to standardize
    
    Returns:
    pd.Series: Standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns=None, outlier_factor=1.5, normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric columns to process
    outlier_factor (float): IQR factor for outlier removal
    normalize_method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if df.empty:
        return df
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            # Remove outliers
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            
            # Normalize data
            if normalize_method == 'minmax':
                cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
            elif normalize_method == 'zscore':
                cleaned_df[f'{col}_standardized'] = standardize_zscore(cleaned_df, col)
            else:
                raise ValueError(f"Unknown normalization method: {normalize_method}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"Dataframe has less than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataframe is valid"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'temperature': [22.5, 23.1, 100.0, 21.8, 22.9, -10.0, 24.2, 23.5],
        'humidity': [45, 48, 52, 43, 47, 200, 49, 46],
        'pressure': [1013, 1012, 1015, 1011, 1014, 500, 1016, 1013]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    print("\nData shape:", df.shape)
    
    # Clean the data
    cleaned = clean_dataset(df, numeric_columns=['temperature', 'humidity', 'pressure'])
    print("\nCleaned data:")
    print(cleaned)
    print("\nCleaned data shape:", cleaned.shape)
    
    # Validate
    is_valid, message = validate_dataframe(cleaned, min_rows=1)
    print(f"\nValidation: {is_valid} - {message}")import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any

def remove_duplicates(df: pd.DataFrame, subset: Union[List[str], None] = None) -> pd.DataFrame:
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
                         type_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Convert columns to specified data types.
    
    Args:
        df: Input DataFrame
        type_mapping: Dictionary mapping column names to target types
    
    Returns:
        DataFrame with converted column types
    """
    df_copy = df.copy()
    
    for column, dtype in type_mapping.items():
        if column in df_copy.columns:
            try:
                if dtype == 'datetime':
                    df_copy[column] = pd.to_datetime(df_copy[column])
                elif dtype == 'numeric':
                    df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
                elif dtype == 'category':
                    df_copy[column] = df_copy[column].astype('category')
                else:
                    df_copy[column] = df_copy[column].astype(dtype)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert column '{column}' to {dtype}: {e}")
    
    return df_copy

def handle_missing_values(df: pd.DataFrame, 
                          strategy: str = 'drop',
                          fill_value: Any = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill values
        fill_value: Value to use when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        return df.fillna(fill_value)
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")

def clean_dataframe(df: pd.DataFrame,
                    deduplicate: bool = True,
                    type_conversions: Union[Dict[str, str], None] = None,
                    missing_strategy: str = 'drop',
                    fill_value: Any = None) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        type_conversions: Column type mapping dictionary
        missing_strategy: Strategy for handling missing values
        fill_value: Value to fill missing values with
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    cleaned_df = handle_missing_values(cleaned_df, missing_strategy, fill_value)
    
    return cleaned_df

def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
        'age': ['25', '30', '30', 'thirty-five', '40', '45'],
        'score': [85.5, 92.0, 92.0, None, 78.5, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation results:")
    print(validate_dataframe(df))
    
    type_map = {
        'id': 'int',
        'age': 'numeric',
        'score': 'float'
    }
    
    cleaned_df = clean_dataframe(
        df,
        deduplicate=True,
        type_conversions=type_map,
        missing_strategy='fill',
        fill_value=0
    )
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned validation results:")
    print(validate_dataframe(cleaned_df))