
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    for column in cleaned_df.select_dtypes(include=[np.number]).columns:
        if cleaned_df[column].isnull().any():
            if missing_strategy == 'mean':
                cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            elif missing_strategy == 'median':
                cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            elif missing_strategy == 'mode':
                cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
            elif missing_strategy == 'drop':
                cleaned_df.dropna(subset=[column], inplace=True)
    
    # Remove outliers using Z-score method
    if outlier_threshold > 0:
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        z_scores = np.abs((cleaned_df[numeric_columns] - cleaned_df[numeric_columns].mean()) / 
                         cleaned_df[numeric_columns].std())
        outlier_mask = (z_scores < outlier_threshold).all(axis=1)
        cleaned_df = cleaned_df[outlier_mask].reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
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
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in the dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): Columns to normalize (None for all numeric columns)
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    normalized_df = df.copy()
    
    if columns is None:
        columns = normalized_df.select_dtypes(include=[np.number]).columns
    
    for column in columns:
        if column in normalized_df.columns and pd.api.types.is_numeric_dtype(normalized_df[column]):
            if method == 'minmax':
                col_min = normalized_df[column].min()
                col_max = normalized_df[column].max()
                if col_max != col_min:
                    normalized_df[column] = (normalized_df[column] - col_min) / (col_max - col_min)
            elif method == 'zscore':
                col_mean = normalized_df[column].mean()
                col_std = normalized_df[column].std()
                if col_std != 0:
                    normalized_df[column] = (normalized_df[column] - col_mean) / col_std
    
    return normalized_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    print("\nCleaned data:")
    cleaned = clean_dataset(df, missing_strategy='mean', outlier_threshold=2)
    print(cleaned)
    
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")
    
    normalized = normalize_data(cleaned, method='minmax')
    print("\nNormalized data:")
    print(normalized)
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.randn(100) * 100,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[10, 'value'] = 1000
    df.loc[20, 'value'] = -800
    
    print("Original DataFrame shape:", df.shape)
    print("Original value statistics:")
    print(df['value'].describe())
    
    cleaned_df = clean_numeric_data(df, columns=['value'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned value statistics:")
    print(cleaned_df['value'].describe())