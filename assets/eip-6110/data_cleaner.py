
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

if __name__ == "__main__":
    sample_data = {'values': [1, 2, 3, 4, 5, 100, 200, 300]}
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("DataFrame after outlier removal:")
    print(cleaned_df)
    print()
    
    stats = calculate_summary_statistics(df, 'values')
    print("Summary statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()
    
    normalized_df = normalize_column(df, 'values', method='minmax')
    print("DataFrame after min-max normalization:")
    print(normalized_df)
import pandas as pd
import numpy as np

def clean_dataset(df, duplicate_threshold=0.8, missing_strategy='median'):
    """
    Clean dataset by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        duplicate_threshold: threshold for considering rows as duplicates (0.0 to 1.0)
        missing_strategy: strategy for handling missing values ('median', 'mean', 'drop', 'fill')
    
    Returns:
        Cleaned pandas DataFrame
    """
    original_shape = df.shape
    
    # Remove exact duplicates
    df = df.drop_duplicates()
    
    # Remove approximate duplicates based on threshold
    if duplicate_threshold < 1.0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            similarity_matrix = df[numeric_cols].corr().abs()
            duplicate_mask = similarity_matrix > duplicate_threshold
            np.fill_diagonal(duplicate_mask.values, False)
            duplicate_pairs = np.where(duplicate_mask)
            
            if len(duplicate_pairs[0]) > 0:
                duplicate_indices = set()
                for i, j in zip(duplicate_pairs[0], duplicate_pairs[1]):
                    if i < j:
                        duplicate_indices.add(j)
                
                df = df.drop(index=list(duplicate_indices)).reset_index(drop=True)
    
    # Handle missing values
    if missing_strategy == 'drop':
        df = df.dropna()
    elif missing_strategy in ['median', 'mean']:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                if missing_strategy == 'median':
                    fill_value = df[col].median()
                else:
                    fill_value = df[col].mean()
                df[col] = df[col].fillna(fill_value)
    elif missing_strategy == 'fill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Remove constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
    
    # Log cleaning results
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df.shape}")
    print(f"Rows removed: {original_shape[0] - df.shape[0]}")
    print(f"Columns removed: {original_shape[1] - df.shape[1]}")
    
    return df

def validate_dataset(df, min_rows=10, required_columns=None):
    """
    Validate dataset meets minimum requirements.
    
    Args:
        df: pandas DataFrame to validate
        min_rows: minimum number of rows required
        required_columns: list of column names that must be present
    
    Returns:
        Boolean indicating if dataset is valid
    """
    if df.shape[0] < min_rows:
        print(f"Dataset has only {df.shape[0]} rows, minimum required is {min_rows}")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            print(f"Column '{col}' contains infinite values")
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'feature_a': [1, 2, 2, 3, 4, 5, None, 7, 8, 9],
        'feature_b': [10, 20, 20, 30, 40, 50, 60, 70, 80, 90],
        'feature_c': [100, 200, 200, 300, 400, 500, 600, 700, 800, 900],
        'constant_col': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    
    df = pd.DataFrame(sample_data)
    cleaned_df = clean_dataset(df, duplicate_threshold=0.95, missing_strategy='median')
    
    is_valid = validate_dataset(cleaned_df, min_rows=5, required_columns=['feature_a', 'feature_b'])
    print(f"Dataset is valid: {is_valid}")