
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
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: multiplier for IQR (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        data[column + '_normalized'] = 0.5
    else:
        data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    
    return data

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        data[column + '_standardized'] = 0
    else:
        data[column + '_standardized'] = (data[column] - mean_val) / std_val
    
    return data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        data: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    result = data.copy()
    
    for col in columns:
        if col not in result.columns:
            continue
            
        if strategy == 'drop':
            result = result.dropna(subset=[col])
        elif strategy == 'mean':
            result[col] = result[col].fillna(result[col].mean())
        elif strategy == 'median':
            result[col] = result[col].fillna(result[col].median())
        elif strategy == 'mode':
            result[col] = result[col].fillna(result[col].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return result

def validate_dataframe(data, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        data: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if data.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    original_shape = df.shape
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Removed {original_shape[0] - cleaned_df.shape[0]} rows")
    return cleaned_df
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
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True