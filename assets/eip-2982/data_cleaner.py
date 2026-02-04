import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def z_score_normalize(data, column):
    """
    Normalize data using z-score normalization
    """
    mean = data[column].mean()
    std = data[column].std()
    
    normalized_data = data.copy()
    normalized_data[column] = (data[column] - mean) / std
    return normalized_data

def min_max_normalize(data, column):
    """
    Normalize data using min-max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    normalized_data = data.copy()
    normalized_data[column] = (data[column] - min_val) / (max_val - min_val)
    return normalized_data

def clean_dataset(df, numeric_columns, outlier_factor=1.5, normalization_method='zscore'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column, outlier_factor)
            
            if normalization_method == 'zscore':
                cleaned_df = z_score_normalize(cleaned_df, column)
            elif normalization_method == 'minmax':
                cleaned_df = min_max_normalize(cleaned_df, column)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate dataset structure and content
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    validation_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': numeric_cols,
        'has_missing_values': df.isnull().any().any(),
        'missing_values_count': df.isnull().sum().sum()
    }
    
    return validation_report
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = "drop",
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    input_path: Path to input CSV file
    output_path: Path to save cleaned CSV file
    missing_strategy: Strategy for handling missing values ('drop', 'fill', 'mean')
    fill_value: Value to use when missing_strategy is 'fill'
    
    Returns:
    Cleaned DataFrame
    """
    
    df = pd.read_csv(input_path)
    
    original_rows = len(df)
    print(f"Original data: {original_rows} rows, {len(df.columns)} columns")
    
    df = df.drop_duplicates()
    
    if missing_strategy == "drop":
        df = df.dropna()
    elif missing_strategy == "fill":
        if fill_value is None:
            raise ValueError("fill_value must be provided when using 'fill' strategy")
        df = df.fillna(fill_value)
    elif missing_strategy == "mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    cleaned_rows = len(df)
    print(f"Cleaned data: {cleaned_rows} rows, {len(df.columns)} columns")
    print(f"Removed {original_rows - cleaned_rows} rows")
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    
    return df

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame for common data quality issues.
    
    Returns:
    True if data passes validation checks
    """
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if df.isnull().sum().sum() > 0:
        print("Warning: DataFrame contains missing values")
        return False
    
    if df.duplicated().sum() > 0:
        print("Warning: DataFrame contains duplicates")
        return False
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data(
        'sample_data.csv',
        'cleaned_sample_data.csv',
        missing_strategy='mean'
    )
    
    is_valid = validate_dataframe(cleaned_df)
    print(f"Data is valid: {is_valid}")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (bool): Whether to fill missing values. Default is False.
        fill_value: Value to use for filling missing data. Default is 0.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
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

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
        multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df