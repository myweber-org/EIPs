import pandas as pd

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for nulls.
            If None, checks all columns. Defaults to None.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove rows with null values
    if columns_to_check:
        cleaned_df = cleaned_df.dropna(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.dropna()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that required columns exist in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if all required columns exist, False otherwise.
    """
    existing_columns = set(df.columns)
    required_set = set(required_columns)
    
    return required_set.issubset(existing_columns)

# Example usage (commented out)
# if __name__ == "__main__":
#     sample_data = {
#         'id': [1, 2, 3, 4, 5, 5],
#         'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],
#         'age': [25, 30, 35, None, 40, 40]
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned = clean_dataset(df)
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     is_valid = validate_data(cleaned, ['id', 'name'])
#     print(f"\nData validation result: {is_valid}")
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, strategy='mean'):
    """
    Clean a CSV file by handling missing values.
    
    Parameters:
    input_path (str): Path to the input CSV file
    output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame
    strategy (str): Method for handling missing values: 'mean', 'median', 'mode', or 'drop'
    
    Returns:
    pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None
    """
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    if strategy == 'drop':
        df_cleaned = df.dropna()
    elif strategy == 'mean':
        df_cleaned = df.fillna(df.select_dtypes(include=[np.number]).mean())
    elif strategy == 'median':
        df_cleaned = df.fillna(df.select_dtypes(include=[np.number]).median())
    elif strategy == 'mode':
        df_cleaned = df.fillna(df.mode().iloc[0])
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'drop', 'mean', 'median', or 'mode'")
    
    new_shape = df_cleaned.shape
    rows_removed = original_shape[0] - new_shape[0] if strategy == 'drop' else 0
    
    print(f"Cleaned data shape: {new_shape}")
    print(f"Rows removed: {rows_removed}")
    print(f"Missing values after cleaning: {df_cleaned.isnull().sum().sum()}")
    
    if output_path:
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return None
    else:
        return df_cleaned

def validate_csv_columns(input_path, required_columns):
    """
    Validate that a CSV file contains required columns.
    
    Parameters:
    input_path (str): Path to the CSV file
    required_columns (list): List of required column names
    
    Returns:
    tuple: (bool, list) - (is_valid, missing_columns)
    """
    
    df = pd.read_csv(input_path)
    existing_columns = set(df.columns)
    required_set = set(required_columns)
    
    missing_columns = list(required_set - existing_columns)
    is_valid = len(missing_columns) == 0
    
    return is_valid, missing_columns

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'temperature': [22.5, np.nan, 24.0, np.nan, 23.0],
        'humidity': [45.0, 50.0, np.nan, 48.0, 47.0],
        'pressure': [1013.2, 1012.8, 1013.5, np.nan, 1013.0]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    print("Testing data cleaning utility...")
    cleaned_df = clean_csv_data('test_data.csv', strategy='mean')
    
    is_valid, missing = validate_csv_columns('test_data.csv', ['id', 'temperature', 'humidity'])
    print(f"CSV validation: {is_valid}, missing columns: {missing}")
    
    import os
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def remove_outliers_zscore(df, columns=None, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
        df_clean = df_clean[(z_scores < threshold) | (df_clean[col].isna())]
    
    return df_clean

def normalize_minmax(df, columns=None):
    """
    Normalize data using Min-Max scaling
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        
        if max_val > min_val:
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    
    return df_normalized

def normalize_zscore(df, columns=None):
    """
    Normalize data using Z-score standardization
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        mean_val = df_normalized[col].mean()
        std_val = df_normalized[col].std()
        
        if std_val > 0:
            df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    
    return df_normalized

def clean_dataset(df, outlier_method='iqr', normalize_method='minmax', outlier_params=None, normalize_params=None):
    """
    Main function to clean dataset with specified methods
    """
    if outlier_params is None:
        outlier_params = {}
    if normalize_params is None:
        normalize_params = {}
    
    df_clean = df.copy()
    
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df_clean, **outlier_params)
    elif outlier_method == 'zscore':
        df_clean = remove_outliers_zscore(df_clean, **outlier_params)
    
    if normalize_method == 'minmax':
        df_clean = normalize_minmax(df_clean, **normalize_params)
    elif normalize_method == 'zscore':
        df_clean = normalize_zscore(df_clean, **normalize_params)
    
    return df_clean

def get_cleaning_summary(original_df, cleaned_df):
    """
    Generate summary statistics before and after cleaning
    """
    summary = {
        'original_rows': len(original_df),
        'cleaned_rows': len(cleaned_df),
        'rows_removed': len(original_df) - len(cleaned_df),
        'removal_percentage': (len(original_df) - len(cleaned_df)) / len(original_df) * 100
    }
    
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        summary[f'{col}_original_mean'] = original_df[col].mean()
        summary[f'{col}_cleaned_mean'] = cleaned_df[col].mean()
        summary[f'{col}_original_std'] = original_df[col].std()
        summary[f'{col}_cleaned_std'] = cleaned_df[col].std()
    
    return summary
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def normalize_minmax(dataframe, columns=None):
    """
    Normalize data using min-max scaling
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            
            if col_max != col_min:
                normalized_df[col] = (dataframe[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0
    
    return normalized_df

def standardize_zscore(dataframe, columns=None):
    """
    Standardize data using z-score normalization
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    standardized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            mean_val = dataframe[col].mean()
            std_val = dataframe[col].std()
            
            if std_val > 0:
                standardized_df[col] = (dataframe[col] - mean_val) / std_val
            else:
                standardized_df[col] = 0
    
    return standardized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    filled_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and dataframe[col].isnull().any():
            if strategy == 'mean':
                fill_value = dataframe[col].mean()
            elif strategy == 'median':
                fill_value = dataframe[col].median()
            elif strategy == 'mode':
                fill_value = dataframe[col].mode()[0]
            elif strategy == 'zero':
                fill_value = 0
            else:
                raise ValueError("Invalid strategy. Use 'mean', 'median', 'mode', or 'zero'")
            
            filled_df[col] = dataframe[col].fillna(fill_value)
    
    return filled_df

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if len(dataframe) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True