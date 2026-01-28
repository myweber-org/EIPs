
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(file_path: str, 
                   missing_strategy: str = 'drop',
                   fill_value: Optional[float] = None) -> pd.DataFrame:
    """
    Load and clean CSV data by handling missing values.
    
    Args:
        file_path: Path to CSV file
        missing_strategy: Strategy for handling missing values 
                         ('drop', 'fill', 'mean')
        fill_value: Value to fill when using 'fill' strategy
    
    Returns:
        Cleaned pandas DataFrame
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if df.empty:
        return df
    
    if missing_strategy == 'drop':
        df_cleaned = df.dropna()
    elif missing_strategy == 'fill':
        if fill_value is None:
            raise ValueError("fill_value must be provided for 'fill' strategy")
        df_cleaned = df.fillna(fill_value)
    elif missing_strategy == 'mean':
        df_cleaned = df.fillna(df.mean(numeric_only=True))
    else:
        raise ValueError(f"Unknown strategy: {missing_strategy}")
    
    return df_cleaned

def remove_outliers_iqr(df: pd.DataFrame, 
                       columns: Optional[list] = None,
                       multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using Interquartile Range method.
    
    Args:
        df: Input DataFrame
        columns: Columns to process (None for all numeric columns)
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filtered = df.copy()
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_filtered = df_filtered[mask]
    
    return df_filtered

def normalize_data(df: pd.DataFrame,
                  columns: Optional[list] = None,
                  method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize numeric columns in DataFrame.
    
    Args:
        df: Input DataFrame
        columns: Columns to normalize (None for all numeric columns)
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val != 0:
                    df_normalized[col] = (df[col] - mean_val) / std_val
            else:
                raise ValueError(f"Unknown normalization method: {method}")
    
    return df_normalized

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame for common data quality issues.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if DataFrame passes validation checks
    """
    if df.empty:
        return False
    
    if df.isnull().all().any():
        return False
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            if df[col].isnull().all():
                return False
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, 20, 30, np.nan, 50],
        'C': [100, 200, 300, 400, 500]
    })
    
    cleaned = clean_csv_data('dummy_path.csv', missing_strategy='mean')
    print("Data cleaning completed")
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: Input pandas DataFrame
        column_mapping: Dictionary to rename columns
        drop_duplicates: Whether to remove duplicate rows
        normalize_text: Whether to normalize text columns
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
    
    return cleaned_df

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    """
    if pd.isna(text):
        return text
    
    normalized = str(text).lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    return normalized

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and non-empty values.
    
    Args:
        df: Input pandas DataFrame
        required_columns: List of columns that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"Warning: DataFrame contains {null_counts.sum()} null values")
    
    return True, "DataFrame is valid"
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
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
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    mask = z_scores < threshold
    
    valid_indices = data[column].dropna().index[mask]
    filtered_data = data.loc[valid_indices]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric columns)
        outlier_method: 'iqr' or 'zscore' (default: 'iqr')
        normalize_method: 'minmax' or 'zscore' (default: 'minmax')
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if normalize_method == 'minmax':
            cleaned_data[f"{column}_normalized"] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f"{column}_standardized"] = normalize_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown normalize method: {normalize_method}")
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan_ratio=0.1):
    """
    Validate dataset structure and quality.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
        allow_nan_ratio: maximum allowed ratio of NaN values per column
    
    Returns:
        tuple: (is_valid, validation_report)
    """
    validation_report = {}
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            validation_report['missing_columns'] = missing_columns
    
    nan_report = {}
    for column in data.columns:
        nan_count = data[column].isna().sum()
        nan_ratio = nan_count / len(data)
        
        if nan_ratio > allow_nan_ratio:
            nan_report[column] = {
                'nan_count': nan_count,
                'nan_ratio': nan_ratio,
                'status': 'high_nan'
            }
        elif nan_count > 0:
            nan_report[column] = {
                'nan_count': nan_count,
                'nan_ratio': nan_ratio,
                'status': 'has_nan'
            }
        else:
            nan_report[column] = {
                'nan_count': 0,
                'nan_ratio': 0.0,
                'status': 'clean'
            }
    
    validation_report['nan_analysis'] = nan_report
    validation_report['shape'] = data.shape
    validation_report['dtypes'] = data.dtypes.to_dict()
    
    is_valid = True
    if 'missing_columns' in validation_report:
        is_valid = False
    if any(info['status'] == 'high_nan' for info in nan_report.values()):
        is_valid = False
    
    return is_valid, validation_report

def get_summary_statistics(data):
    """
    Generate comprehensive summary statistics.
    
    Args:
        data: pandas DataFrame
    
    Returns:
        DataFrame with summary statistics
    """
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return pd.DataFrame()
    
    summary = pd.DataFrame({
        'mean': numeric_data.mean(),
        'median': numeric_data.median(),
        'std': numeric_data.std(),
        'min': numeric_data.min(),
        'max': numeric_data.max(),
        'q1': numeric_data.quantile(0.25),
        'q3': numeric_data.quantile(0.75),
        'iqr': numeric_data.quantile(0.75) - numeric_data.quantile(0.25),
        'skewness': numeric_data.skew(),
        'kurtosis': numeric_data.kurtosis(),
        'nan_count': numeric_data.isna().sum(),
        'nan_ratio': numeric_data.isna().sum() / len(numeric_data)
    })
    
    return summary.T