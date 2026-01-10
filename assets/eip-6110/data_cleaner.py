
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
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

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from specified numeric columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_columns (list): List of numeric column names to clean.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print(f"Original dataset shape: {df.shape}")
    cleaned_df = clean_dataset(df, ['A', 'B'])
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Outliers removed: {len(df) - len(cleaned_df)}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
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
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
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
        'temperature': [22, 23, 24, 25, 26, 100, 27, 28, 29, -10],
        'humidity': [45, 46, 47, 48, 49, 50, 200, 51, 52, -5],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 2000, 1020, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print(f"Original shape: {df.shape}")
    
    cleaned_df = clean_numeric_data(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print(f"Cleaned shape: {cleaned_df.shape}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
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
    
    return filtered_df.copy()

def remove_outliers_zscore(dataframe, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(dataframe[column].dropna()))
    
    filtered_indices = np.where(z_scores < threshold)[0]
    filtered_df = dataframe.iloc[filtered_indices].copy()
    
    return filtered_df

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using Min-Max scaling.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    df_copy = dataframe.copy()
    
    if columns is None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df_copy.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if not np.issubdtype(df_copy[col].dtype, np.number):
            continue
        
        col_min = df_copy[col].min()
        col_max = df_copy[col].max()
        
        if col_max != col_min:
            df_copy[col] = (df_copy[col] - col_min) / (col_max - col_min)
        else:
            df_copy[col] = 0
    
    return df_copy

def normalize_zscore(dataframe, columns=None):
    """
    Normalize specified columns using Z-score standardization.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns
    """
    df_copy = dataframe.copy()
    
    if columns is None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df_copy.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if not np.issubdtype(df_copy[col].dtype, np.number):
            continue
        
        col_mean = df_copy[col].mean()
        col_std = df_copy[col].std()
        
        if col_std != 0:
            df_copy[col] = (df_copy[col] - col_mean) / col_std
        else:
            df_copy[col] = 0
    
    return df_copy

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns (list): List of column names to process. If None, process all columns.
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    df_copy = dataframe.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    for col in columns:
        if col not in df_copy.columns:
            continue
        
        if df_copy[col].isnull().sum() == 0:
            continue
        
        if strategy == 'drop':
            df_copy = df_copy.dropna(subset=[col])
        elif strategy == 'mean':
            if np.issubdtype(df_copy[col].dtype, np.number):
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        elif strategy == 'median':
            if np.issubdtype(df_copy[col].dtype, np.number):
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        elif strategy == 'mode':
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_copy

def clean_dataset(dataframe, outlier_method='iqr', normalize_method='minmax', 
                  missing_strategy='mean', outlier_columns=None, normalize_columns=None):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    outlier_method (str): Outlier removal method ('iqr', 'zscore', or None)
    normalize_method (str): Normalization method ('minmax', 'zscore', or None)
    missing_strategy (str): Strategy for handling missing values
    outlier_columns (list): Columns for outlier removal
    normalize_columns (list): Columns for normalization
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = dataframe.copy()
    
    # Handle missing values
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy)
    
    # Remove outliers
    if outlier_method:
        if outlier_columns is None:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            outlier_columns = list(numeric_cols)
        
        for col in outlier_columns:
            if col in df_clean.columns and np.issubdtype(df_clean[col].dtype, np.number):
                if outlier_method == 'iqr':
                    df_clean = remove_outliers_iqr(df_clean, col)
                elif outlier_method == 'zscore':
                    df_clean = remove_outliers_zscore(df_clean, col)
    
    # Normalize data
    if normalize_method:
        if normalize_method == 'minmax':
            df_clean = normalize_minmax(df_clean, normalize_columns)
        elif normalize_method == 'zscore':
            df_clean = normalize_zscore(df_clean, normalize_columns)
    
    return df_clean
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Columns to check for duplicates.
            If None, checks all columns. Defaults to None.
        fill_missing (bool, optional): Whether to fill missing values with column mean.
            Defaults to True.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check is None:
        columns_to_check = cleaned_df.columns.tolist()
    
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check, keep='first')
    duplicates_removed = initial_rows - len(cleaned_df)
    
    # Handle missing values
    missing_before = cleaned_df.isnull().sum().sum()
    
    if fill_missing and missing_before > 0:
        for column in cleaned_df.select_dtypes(include=['float64', 'int64']).columns:
            if cleaned_df[column].isnull().any():
                column_mean = cleaned_df[column].mean()
                cleaned_df[column] = cleaned_df[column].fillna(column_mean)
    
    missing_after = cleaned_df.isnull().sum().sum()
    
    # Print cleaning summary
    print(f"Cleaning Summary:")
    print(f"  - Duplicates removed: {duplicates_removed}")
    print(f"  - Missing values before: {missing_before}")
    print(f"  - Missing values after: {missing_after}")
    print(f"  - Final dataset shape: {cleaned_df.shape}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     sample_data = {
#         'id': [1, 2, 2, 3, 4, 5],
#         'value': [10.5, 20.3, 20.3, None, 40.1, 50.0],
#         'category': ['A', 'B', 'B', 'C', None, 'A']
#     }
#     
#     df = pd.DataFrame(sample_data)
#     
#     if validate_dataframe(df):
#         cleaned = clean_dataset(df, columns_to_check=['id', 'category'])
#         print("\nCleaned DataFrame:")
#         print(cleaned)