
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_clean = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            elif strategy == 'drop':
                data_clean = data_clean.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_clean[col] = data_clean[col].fillna(fill_value)
    
    return data_clean

def validate_dataframe(data, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if len(data) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return Trueimport pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Column names to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
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
    removed_count = original_shape[0] - cleaned_df.shape[0]
    print(f"Removed {removed_count} outliers from dataset")
    return cleaned_df
import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame using specified strategy.
    
    Args:
        df: pandas DataFrame
        strategy: Method for handling missing values ('mean', 'median', 'mode', 'drop')
        columns: List of columns to apply cleaning to (None for all columns)
    
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].isnull().sum() == 0:
            continue
        
        if strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        elif strategy == 'mode':
            mode_value = df_clean[col].mode()
            if not mode_value.empty:
                df_clean[col].fillna(mode_value[0], inplace=True)
        
        elif strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
    
    return df_clean

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a specific column.
    
    Args:
        df: pandas DataFrame
        column: Column name to check for outliers
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold value for outlier detection
    
    Returns:
        Boolean mask indicating outliers
    """
    if column not in df.columns:
        return pd.Series([False] * len(df))
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val == 0:
            return pd.Series([False] * len(df))
        z_scores = np.abs((df[column] - mean_val) / std_val)
        return z_scores > threshold
    
    return pd.Series([False] * len(df))

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df: pandas DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax' or 'standard')
    
    Returns:
        Series with normalized values
    """
    if column not in df.columns:
        return pd.Series()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val == min_val:
            return pd.Series([0] * len(df))
        return (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'standard':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val == 0:
            return pd.Series([0] * len(df))
        return (df[column] - mean_val) / std_val
    
    return df[column].copy()

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
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