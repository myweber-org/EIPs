
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fill_missing: Strategy for filling missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Removed rows with missing values")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                else:  # mode
                    fill_value = cleaned_df[col].mode()[0]
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in '{col}' with {fill_missing}: {fill_value:.2f}")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
            print(f"Filled missing values in '{col}' with 'Unknown'")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, message)
    """
    if len(df) < min_rows:
        return False, f"DataFrame has only {len(df)} rows, minimum required is {min_rows}"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if df.isnull().all().any():
        empty_cols = df.columns[df.isnull().all()].tolist()
        return False, f"Columns with all missing values: {empty_cols}"
    
    return True, "Data validation passed"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Args:
        df: DataFrame containing the data
        column: Column name to check for outliers
        method: 'iqr' for interquartile range or 'zscore' for standard deviations
        threshold: Threshold value for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric for outlier detection")
    
    data = df[column].copy()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
    else:  # zscore
        mean = data.mean()
        std = data.std()
        z_scores = np.abs((data - mean) / std)
        mask = z_scores <= threshold
    
    initial_count = len(df)
    filtered_df = df[mask].copy()
    removed_count = initial_count - len(filtered_df)
    
    if removed_count > 0:
        print(f"Removed {removed_count} outliers from column '{column}' using {method} method")
    
    return filtered_df