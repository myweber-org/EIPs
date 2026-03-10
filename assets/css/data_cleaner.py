
import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def remove_outliers(data, column, threshold=1.5):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, column, strategy='mean'):
    if strategy == 'mean':
        fill_value = data[column].mean()
    elif strategy == 'median':
        fill_value = data[column].median()
    elif strategy == 'mode':
        fill_value = data[column].mode()[0]
    else:
        fill_value = 0
    filled_data = data[column].fillna(fill_value)
    return filled_data

def clean_dataset(data, numeric_columns, outlier_threshold=1.5, normalize=True, standardize=False, missing_strategy='mean'):
    cleaned_data = data.copy()
    for col in numeric_columns:
        cleaned_data = remove_outliers(cleaned_data, col, outlier_threshold)
        cleaned_data[col] = handle_missing_values(cleaned_data, col, missing_strategy)
        if normalize:
            cleaned_data[f'{col}_normalized'] = normalize_minmax(cleaned_data, col)
        if standardize:
            cleaned_data[f'{col}_standardized'] = standardize_zscore(cleaned_data, col)
    return cleaned_data
import pandas as pd
import numpy as np
from typing import Optional, List, Union

def clean_dataset(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    fill_missing: Optional[Union[str, float]] = None,
    convert_types: bool = False,
    columns_to_clean: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Clean a pandas DataFrame by handling duplicates, missing values, and type conversion.
    
    Parameters:
    df: Input DataFrame
    drop_duplicates: Whether to remove duplicate rows
    fill_missing: Strategy for filling missing values ('mean', 'median', 'mode', or scalar)
    convert_types: Whether to convert columns to optimal data types
    columns_to_clean: Specific columns to clean (None for all columns)
    
    Returns:
    Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if columns_to_clean is None:
        columns_to_clean = cleaned_df.columns.tolist()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - len(cleaned_df)
        if removed > 0:
            print(f"Removed {removed} duplicate rows")
    
    if fill_missing is not None:
        for col in columns_to_clean:
            if col in cleaned_df.columns and cleaned_df[col].isnull().any():
                if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                elif fill_missing == 'mode':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
                elif isinstance(fill_missing, (int, float)):
                    cleaned_df[col] = cleaned_df[col].fillna(fill_missing)
    
    if convert_types:
        for col in columns_to_clean:
            if col in cleaned_df.columns:
                if pd.api.types.is_object_dtype(cleaned_df[col]):
                    try:
                        cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='ignore')
                        if not pd.api.types.is_datetime64_any_dtype(cleaned_df[col]):
                            cleaned_df[col] = cleaned_df[col].astype('category')
                    except:
                        cleaned_df[col] = cleaned_df[col].astype('category')
                elif pd.api.types.is_integer_dtype(cleaned_df[col]):
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], downcast='integer')
                elif pd.api.types.is_float_dtype(cleaned_df[col]):
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], downcast='float')
    
    return cleaned_df

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains required columns and has no completely empty columns.
    
    Parameters:
    df: DataFrame to validate
    required_columns: List of column names that must be present
    
    Returns:
    Boolean indicating if validation passed
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    empty_columns = df.columns[df.isnull().all()].tolist()
    if empty_columns:
        print(f"Warning: Found completely empty columns: {empty_columns}")
    
    return True

def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str],
    multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df: Input DataFrame
    columns: Numeric columns to process for outliers
    multiplier: IQR multiplier (default 1.5)
    
    Returns:
    DataFrame with outliers removed
    """
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    for col in columns:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    removed = initial_count - len(df_clean)
    if removed > 0:
        print(f"Removed {removed} rows containing outliers")
    
    return df_clean.reset_index(drop=True)