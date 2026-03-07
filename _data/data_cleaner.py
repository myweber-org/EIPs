
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

def handle_missing_values(df, column, strategy='mean'):
    """
    Handle missing values in a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if strategy == 'mean':
        fill_value = df_copy[column].mean()
    elif strategy == 'median':
        fill_value = df_copy[column].median()
    elif strategy == 'mode':
        fill_value = df_copy[column].mode()[0]
    elif strategy == 'drop':
        df_copy = df_copy.dropna(subset=[column])
        return df_copy
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    df_copy[column] = df_copy[column].fillna(fill_value)
    return df_copy
import re
import pandas as pd
from typing import Union, List, Optional

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names: lowercase, replace spaces with underscores.
    """
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def validate_email(email: str) -> bool:
    """
    Validate email format using regex.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def fill_missing_values(df: pd.DataFrame, column: str, value: Union[str, int, float]) -> pd.DataFrame:
    """
    Fill missing values in a specified column with a given value.
    """
    df[column] = df[column].fillna(value)
    return df

def convert_to_datetime(df: pd.DataFrame, column: str, format: str = '%Y-%m-%d') -> pd.DataFrame:
    """
    Convert a column to datetime format.
    """
    df[column] = pd.to_datetime(df[column], format=format, errors='coerce')
    return df

def filter_by_threshold(df: pd.DataFrame, column: str, threshold: float, keep: str = 'above') -> pd.DataFrame:
    """
    Filter rows based on a numeric threshold.
    """
    if keep == 'above':
        return df[df[column] > threshold]
    elif keep == 'below':
        return df[df[column] < threshold]
    else:
        raise ValueError("keep parameter must be 'above' or 'below'")

def main():
    """
    Example usage of data cleaning functions.
    """
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', None],
        'Email': ['alice@example.com', 'invalid-email', 'charlie@test.org', 'david@domain.net'],
        'Score': [85.5, 92.0, 78.5, 88.0],
        'Join Date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05']
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    df = clean_column_names(df)
    df = fill_missing_values(df, 'name', 'Unknown')
    df = convert_to_datetime(df, 'join_date')
    df = filter_by_threshold(df, 'score', 80.0, keep='above')
    
    print("\nCleaned DataFrame:")
    print(df)
    
    email_check = df['email'].apply(validate_email)
    print("\nValid emails:", df[email_check]['email'].tolist())

if __name__ == '__main__':
    main()
import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None):
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        df_cleaned = df.drop_duplicates()
        final_count = len(df_cleaned)
        
        if output_file is None:
            output_file = input_file.replace('.csv', '_cleaned.csv')
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Removed {initial_count - final_count} duplicate rows.")
        print(f"Original rows: {initial_count}")
        print(f"Cleaned rows: {final_count}")
        print(f"Saved to: {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file.csv> [output_file.csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fillna_method=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default True.
        fillna_method (str or None): Method to fill missing values. 
            Options: 'ffill', 'bfill', 'mean', 'median', or None to drop rows.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fillna_method is None:
        cleaned_df = cleaned_df.dropna()
    elif fillna_method in ['ffill', 'bfill']:
        cleaned_df = cleaned_df.fillna(method=fillna_method)
    elif fillna_method == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fillna_method == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    
    # Remove duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'A': [1, 2, None, 4, 4],
#         'B': [5, None, 7, 8, 8],
#         'C': [9, 10, 11, 12, 12]
#     }
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned = clean_dataframe(df, fillna_method='mean')
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     is_valid, message = validate_dataframe(cleaned, ['A', 'B', 'C'])
#     print(f"\nValidation: {is_valid} - {message}")import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows.")
    
    if fill_missing:
        missing_before = df.isnull().sum().sum()
        
        if fill_missing == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_missing == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_missing == 'mode':
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
        elif fill_missing == 'drop':
            df = df.dropna()
        
        missing_after = df.isnull().sum().sum()
        print(f"Handled {missing_before - missing_after} missing values.")
    
    print(f"Dataset cleaned. Original shape: {original_shape}, New shape: {df.shape}")
    return df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the dataset structure and content.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has fewer than {min_rows} rows.")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None, 4],
        'B': [5, None, 5, 6, 7, 8],
        'C': ['x', 'y', 'x', 'z', None, 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df.copy(), drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_data(cleaned_df, required_columns=['A', 'B'], min_rows=3)
        print("\nData validation passed.")
    except ValueError as e:
        print(f"\nData validation failed: {e}")