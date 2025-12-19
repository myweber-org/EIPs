
import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing duplicate rows,
    standardizing column names, and filling missing values.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()

    # Standardize column names: lowercase and replace spaces with underscores
    df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')

    # Fill missing numeric values with column median
    for col in df_cleaned.select_dtypes(include=[np.number]).columns:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    # Fill missing categorical values with mode
    for col in df_cleaned.select_dtypes(include=['object']).columns:
        mode_value = df_cleaned[col].mode()
        if not mode_value.empty:
            df_cleaned[col] = df_cleaned[col].fillna(mode_value.iloc[0])
        else:
            df_cleaned[col] = df_cleaned[col].fillna('unknown')

    return df_cleaned

def validate_dataframe(df):
    """
    Perform basic validation on the DataFrame.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    required_columns = ['id', 'name']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if df['id'].duplicated().any():
        raise ValueError("Duplicate IDs found in the DataFrame")

    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'ID': [1, 2, 2, 3, 4],
        'Name': ['Alice', 'Bob', 'Bob', 'Charlie', None],
        'Age': [25, 30, 30, None, 35],
        'Score': [85.5, 92.0, 92.0, 78.5, None]
    }

    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataframe(df)
    print(cleaned_df)

    try:
        validate_dataframe(cleaned_df)
        print("\nData validation passed.")
    except ValueError as e:
        print(f"\nData validation failed: {e}")
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
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

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
    return data[mask]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame or Series
        column: column name to normalize
    
    Returns:
        Normalized data
    """
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        col_data = data[column]
    else:
        col_data = data
    
    min_val = col_data.min()
    max_val = col_data.max()
    
    if max_val == min_val:
        return col_data * 0  # Return zeros if all values are same
    
    return (col_data - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame or Series
        column: column name to normalize
    
    Returns:
        Standardized data
    """
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        col_data = data[column]
    else:
        col_data = data
    
    mean_val = col_data.mean()
    std_val = col_data.std()
    
    if std_val == 0:
        return col_data * 0  # Return zeros if no variance
    
    return (col_data - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_method: 'iqr', 'zscore', or None
        normalize_method: 'minmax', 'zscore', or None
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    # Remove outliers
    if outlier_method:
        for col in numeric_columns:
            if col in cleaned_data.columns:
                if outlier_method == 'iqr':
                    cleaned_data = remove_outliers_iqr(cleaned_data, col)
                elif outlier_method == 'zscore':
                    cleaned_data = remove_outliers_zscore(cleaned_data, col)
    
    # Normalize data
    if normalize_method:
        for col in numeric_columns:
            if col in cleaned_data.columns:
                if normalize_method == 'minmax':
                    cleaned_data[col] = normalize_minmax(cleaned_data, col)
                elif normalize_method == 'zscore':
                    cleaned_data[col] = normalize_zscore(cleaned_data, col)
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=True, nan_threshold=0.5):
    """
    Validate data quality.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required columns
        allow_nan: whether to allow NaN values
        nan_threshold: maximum allowed NaN ratio per column
    
    Returns:
        tuple: (is_valid, issues)
    """
    issues = []
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
    
    # Check NaN values
    if not allow_nan:
        nan_cols = data.columns[data.isna().any()].tolist()
        if nan_cols:
            issues.append(f"Columns with NaN values: {nan_cols}")
    else:
        for col in data.columns:
            nan_ratio = data[col].isna().mean()
            if nan_ratio > nan_threshold:
                issues.append(f"Column '{col}' has {nan_ratio:.1%} NaN values")
    
    # Check numeric columns for infinite values
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(data[col])):
            issues.append(f"Column '{col}' contains infinite values")
    
    is_valid = len(issues) == 0
    return is_valid, issues