
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_column(dataframe, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to normalize
        method: 'minmax' or 'zscore' normalization
    
    Returns:
        DataFrame with normalized column added as '{column}_normalized'
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result_df = dataframe.copy()
    
    if method == 'minmax':
        min_val = result_df[column].min()
        max_val = result_df[column].max()
        
        if max_val == min_val:
            result_df[f'{column}_normalized'] = 0.5
        else:
            result_df[f'{column}_normalized'] = (result_df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = result_df[column].mean()
        std_val = result_df[column].std()
        
        if std_val == 0:
            result_df[f'{column}_normalized'] = 0
        else:
            result_df[f'{column}_normalized'] = (result_df[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return result_df

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5, normalize=True):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: Input DataFrame
        numeric_columns: List of numeric columns to process (default: all numeric)
        outlier_threshold: IQR threshold for outlier removal
        normalize: Whether to normalize columns
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            # Remove outliers
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
            
            # Normalize if requested
            if normalize:
                cleaned_df = normalize_column(cleaned_df, col, method='minmax')
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(dataframe, required_columns=None, allow_nan=False):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: DataFrame to validate
        required_columns: List of required column names
        allow_nan: Whether NaN values are allowed
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if dataframe.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if not allow_nan and dataframe.isnull().any().any():
        nan_cols = dataframe.columns[dataframe.isnull().any()].tolist()
        return False, f"NaN values found in columns: {nan_cols}"
    
    return True, "DataFrame is valid"import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_method (str or None): Method to fill missing values ('mean', 'median', 'mode', or None).
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method is not None:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        if fill_method == 'mean':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        elif fill_method == 'median':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
        elif fill_method == 'mode':
            for col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
        else:
            raise ValueError("fill_method must be 'mean', 'median', 'mode', or None")
    
    # Drop duplicates
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {},
        'dtypes': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['is_valid'] = False
            validation_results['missing_columns'] = missing
    
    # Count null values per column
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            validation_results['null_counts'][col] = null_count
    
    # Get data types
    validation_results['dtypes'] = df.dtypes.to_dict()
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, None, 4, 1],
        'B': [5, None, 7, 8, 5],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation results:")
    print(validate_dataframe(df))
    
    cleaned = clean_dataset(df, fill_method='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)