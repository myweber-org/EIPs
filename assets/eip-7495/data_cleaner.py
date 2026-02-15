
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
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
    Standardize data using Z-score normalization.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values using specified strategy.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mean())
    elif strategy == 'median':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
    elif strategy == 'mode':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mode()[0])
    elif strategy == 'drop':
        data = data.dropna()
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    return data

def detect_skewness(data, column, threshold=0.5):
    """
    Detect skewness in data column.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    skewness = stats.skew(data[column].dropna())
    is_skewed = abs(skewness) > threshold
    
    return skewness, is_skewed

def log_transform(data, column):
    """
    Apply log transformation to reduce skewness.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if data[column].min() <= 0:
        transformed = np.log1p(data[column] - data[column].min() + 1)
    else:
        transformed = np.log(data[column])
    
    return transformed
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using Interquartile Range method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
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
    Standardize data using Z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_factor=1.5):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    removal_stats = {}
    
    for col in numeric_columns:
        if col in df.columns:
            cleaned_df, removed = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            removal_stats[col] = removed
            
            cleaned_df[f"{col}_normalized"] = normalize_minmax(cleaned_df, col)
            cleaned_df[f"{col}_standardized"] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df, removal_stats

def validate_data(df, required_columns, numeric_threshold=0.8):
    """
    Validate data quality and completeness
    """
    validation_results = {
        'missing_columns': [],
        'high_missing_values': [],
        'low_variance': []
    }
    
    for col in required_columns:
        if col not in df.columns:
            validation_results['missing_columns'].append(col)
        else:
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > 0.3:
                validation_results['high_missing_values'].append((col, missing_ratio))
            
            if df[col].dtype in [np.float64, np.int64]:
                if df[col].std() < numeric_threshold:
                    validation_results['low_variance'].append((col, df[col].std()))
    
    return validation_results
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe: Input pandas DataFrame.
        subset: Column label or sequence of labels to consider for duplicates.
                If None, all columns are used.
        keep: Determines which duplicates to mark.
              'first' : Mark duplicates as False except for the first occurrence.
              'last' : Mark duplicates as False except for the last occurrence.
              False : Mark all duplicates as True.
    
    Returns:
        DataFrame with duplicates removed.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_missing_values(dataframe, strategy='drop', fill_value=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        dataframe: Input pandas DataFrame.
        strategy: How to handle missing values.
                  'drop': Remove rows with any missing values.
                  'fill': Fill missing values with specified fill_value.
        fill_value: Value to use when strategy is 'fill'.
    
    Returns:
        DataFrame with missing values handled.
    """
    if strategy == 'drop':
        cleaned_df = dataframe.dropna()
    elif strategy == 'fill':
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy is 'fill'")
        cleaned_df = dataframe.fillna(fill_value)
    else:
        raise ValueError("strategy must be either 'drop' or 'fill'")
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: Input pandas DataFrame.
        required_columns: List of column names that must be present.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if dataframe.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"