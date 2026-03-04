
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    return filtered_df

def remove_outliers_zscore(dataframe, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        threshold: Z-score threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(dataframe[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    
    column_indices = dataframe[column].dropna().index
    valid_indices = column_indices[filtered_indices]
    
    filtered_df = dataframe.loc[valid_indices]
    return filtered_df

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using Min-Max scaling.
    
    Args:
        dataframe: pandas DataFrame
        columns: List of columns to normalize. If None, normalize all numeric columns.
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            
            if col_max != col_min:
                normalized_df[col] = (dataframe[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0
    
    return normalized_df

def normalize_zscore(dataframe, columns=None):
    """
    Normalize specified columns using Z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        columns: List of columns to normalize. If None, normalize all numeric columns.
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            col_mean = dataframe[col].mean()
            col_std = dataframe[col].std()
            
            if col_std != 0:
                normalized_df[col] = (dataframe[col] - col_mean) / col_std
            else:
                normalized_df[col] = 0
    
    return normalized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        dataframe: pandas DataFrame
        strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: List of columns to process. If None, process all columns.
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns
    
    processed_df = dataframe.copy()
    
    for col in columns:
        if col not in processed_df.columns:
            continue
            
        if processed_df[col].isnull().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(processed_df[col]):
                fill_value = processed_df[col].mean()
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(processed_df[col]):
                fill_value = processed_df[col].median()
            elif strategy == 'mode':
                fill_value = processed_df[col].mode()[0] if not processed_df[col].mode().empty else None
            elif strategy == 'drop':
                processed_df = processed_df.dropna(subset=[col])
                continue
            else:
                fill_value = None
            
            if fill_value is not None:
                processed_df[col] = processed_df[col].fillna(fill_value)
    
    return processed_df

def clean_dataset(dataframe, outlier_method='iqr', normalization_method='minmax', 
                  missing_strategy='mean', outlier_columns=None, normalize_columns=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: Input pandas DataFrame
        outlier_method: 'iqr', 'zscore', or None
        normalization_method: 'minmax', 'zscore', or None
        missing_strategy: 'mean', 'median', 'mode', or 'drop'
        outlier_columns: Columns for outlier removal
        normalize_columns: Columns for normalization
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = dataframe.copy()
    
    if missing_strategy:
        cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if outlier_method and outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                if outlier_method == 'iqr':
                    cleaned_df = remove_outliers_iqr(cleaned_df, col)
                elif outlier_method == 'zscore':
                    cleaned_df = remove_outliers_zscore(cleaned_df, col)
    
    if normalization_method and normalize_columns:
        if normalization_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, normalize_columns)
        elif normalization_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, normalize_columns)
    
    return cleaned_df
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_dataimport pandas as pd
import numpy as np

def clean_dataset(df, numeric_columns=None, method='median', outlier_threshold=3):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    method (str): Imputation method ('mean', 'median', 'mode')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    
    if df.empty:
        return df
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    # Handle missing values
    for col in numeric_columns:
        if col in df_clean.columns:
            if df_clean[col].isnull().any():
                if method == 'mean':
                    fill_value = df_clean[col].mean()
                elif method == 'median':
                    fill_value = df_clean[col].median()
                elif method == 'mode':
                    fill_value = df_clean[col].mode()[0]
                else:
                    fill_value = df_clean[col].median()
                
                df_clean[col].fillna(fill_value, inplace=True)
    
    # Remove outliers using Z-score method
    z_scores = np.abs((df_clean[numeric_columns] - df_clean[numeric_columns].mean()) / 
                      df_clean[numeric_columns].std())
    
    outlier_mask = (z_scores < outlier_threshold).all(axis=1)
    df_clean = df_clean[outlier_mask].reset_index(drop=True)
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    
    if df.empty:
        return False, "Dataframe is empty"
    
    if len(df) < min_rows:
        return False, f"Dataframe has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize numeric columns in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): Columns to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_normalized = df.copy()
    
    for col in columns:
        if col in df_normalized.columns and df_normalized[col].dtype in [np.float64, np.int64]:
            if method == 'minmax':
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df_normalized[col].mean()
                std_val = df_normalized[col].std()
                if std_val > 0:
                    df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    
    return df_normalized

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'feature1': [1, 2, np.nan, 4, 100],
        'feature2': [5, 6, 7, np.nan, 8],
        'category': ['A', 'B', 'A', 'B', 'A']
    }
    
    df_sample = pd.DataFrame(sample_data)
    print("Original data:")
    print(df_sample)
    
    # Clean the data
    df_cleaned = clean_dataset(df_sample, method='median')
    print("\nCleaned data:")
    print(df_cleaned)
    
    # Validate the cleaned data
    is_valid, message = validate_data(df_cleaned, min_rows=2)
    print(f"\nValidation: {is_valid} - {message}")
    
    # Normalize the cleaned data
    df_normalized = normalize_data(df_cleaned, method='minmax')
    print("\nNormalized data:")
    print(df_normalized)