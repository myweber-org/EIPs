
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Strategy to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
        print("Filled missing categorical values with mode")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append("Input is not a pandas DataFrame")
        return validation_results
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        validation_results['warnings'].append("DataFrame is empty")
    
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        validation_results['warnings'].append(f"Found {null_counts.sum()} missing values")
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'x', 'y', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaning with mean imputation...")
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print("\nValidation Results:")
    print(validation)import numpy as np
import pandas as pd

def normalize_data(data, method='minmax'):
    """
    Normalize data using specified method.
    
    Args:
        data: Input data array
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        Normalized data array
    """
    if method == 'minmax':
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min == 0:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    
    elif method == 'zscore':
        data_mean = np.mean(data)
        data_std = np.std(data)
        if data_std == 0:
            return np.zeros_like(data)
        return (data - data_mean) / data_std
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")

def remove_outliers_iqr(data, multiplier=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: Input data array
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        Data with outliers removed
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return data[(data >= lower_bound) & (data <= upper_bound)]

def clean_dataset(df, columns=None, normalize=True, remove_outliers=True):
    """
    Clean dataset by normalizing and removing outliers.
    
    Args:
        df: Input DataFrame
        columns: Columns to process (None for all numeric columns)
        normalize: Whether to normalize data
        remove_outliers: Whether to remove outliers
    
    Returns:
        Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    result_df = df.copy()
    
    for column in columns:
        if column not in df.columns:
            continue
            
        column_data = df[column].values
        
        if remove_outliers:
            column_data = remove_outliers_iqr(column_data)
        
        if normalize and len(column_data) > 0:
            column_data = normalize_data(column_data)
        
        result_df[column] = pd.Series(column_data, index=result_df.index[:len(column_data)])
    
    return result_df

def calculate_statistics(data):
    """
    Calculate basic statistics for data.
    
    Args:
        data: Input data array
    
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'count': len(data)
    }

def validate_data(data, allow_nan=False):
    """
    Validate data for common issues.
    
    Args:
        data: Input data array
        allow_nan: Whether to allow NaN values
    
    Returns:
        Boolean indicating if data is valid
    """
    if not isinstance(data, (np.ndarray, list, pd.Series)):
        return False
    
    data_array = np.array(data)
    
    if not allow_nan and np.any(np.isnan(data_array)):
        return False
    
    if len(data_array) == 0:
        return False
    
    return True