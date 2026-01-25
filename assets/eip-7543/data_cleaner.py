import pandas as pd
import numpy as np

def load_and_clean_csv(file_path, missing_strategy='drop'):
    """
    Load a CSV file and perform basic cleaning operations.
    
    Args:
        file_path (str): Path to the CSV file.
        missing_strategy (str): Strategy for handling missing values.
            'drop' - Drop rows with any missing values.
            'fill_mean' - Fill numeric columns with mean, categorical with mode.
            'fill_median' - Fill numeric columns with median, categorical with mode.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")
    
    # Remove duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    
    # Handle missing values
    if missing_strategy == 'drop':
        df = df.dropna()
    elif missing_strategy in ['fill_mean', 'fill_median']:
        for column in df.columns:
            if df[column].dtype in [np.float64, np.int64]:
                if missing_strategy == 'fill_mean':
                    fill_value = df[column].mean()
                else:
                    fill_value = df[column].median()
                df[column] = df[column].fillna(fill_value)
            else:
                # For categorical columns, fill with mode
                mode_value = df[column].mode()
                if not mode_value.empty:
                    df[column] = df[column].fillna(mode_value.iloc[0])
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    print(f"Data cleaning completed:")
    print(f"  - Removed {duplicates_removed} duplicate rows")
    print(f"  - Final dataset has {len(df)} rows and {len(df.columns)} columns")
    
    return df

def normalize_numeric_columns(df, columns=None):
    """
    Normalize numeric columns to range [0, 1].
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        columns (list): List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
        pandas.DataFrame: DataFrame with normalized columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for column in columns:
        if column in df.columns and df[column].dtype in [np.float64, np.int64]:
            col_min = df[column].min()
            col_max = df[column].max()
            
            if col_max != col_min:
                df[column] = (df[column] - col_min) / (col_max - col_min)
            else:
                df[column] = 0
    
    return df

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        columns (list): List of column names to check for outliers.
        threshold (float): Multiplier for IQR (default 1.5).
    
    Returns:
        pandas.DataFrame: DataFrame with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    initial_rows = len(df)
    
    for column in columns:
        if column in df.columns and df[column].dtype in [np.float64, np.int64]:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    outliers_removed = initial_rows - len(df)
    print(f"Removed {outliers_removed} rows containing outliers")
    
    return df.reset_index(drop=True)

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pandas.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        bool: True if validation passes, raises exception otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

# Example usage
if __name__ == "__main__":
    # This is just for demonstration purposes
    sample_data = {
        'A': [1, 2, 3, 4, 5, 5, 6, None, 8, 9],
        'B': [10, 20, 30, 40, 50, 50, 60, 70, 80, 90],
        'C': ['a', 'b', 'c', 'd', 'e', 'e', 'f', 'g', 'h', 'i']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = load_and_clean_csv('dummy_path', missing_strategy='fill_mean')
    print("Cleaned DataFrame:")
    print(cleaned_df)