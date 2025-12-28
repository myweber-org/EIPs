import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {len(df) - len(cleaned_df)} duplicate rows.")
    
    if fill_missing == 'drop':
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna()
        print(f"Dropped {initial_rows - len(cleaned_df)} rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                cleaned_df[column].fillna(fill_value, inplace=True)
                print(f"Filled missing values in column '{column}' with {fill_missing}: {fill_value}")
    
    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, np.nan],
        'B': [10, 20, 20, 40, np.nan, 60],
        'C': ['x', 'y', 'y', 'z', 'z', 'x']
    }
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print("\nCleaning data...")
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    print("\nValidating cleaned data...")
    is_valid = validate_data(cleaned, required_columns=['A', 'B', 'C'], min_rows=1)
    print(f"Validation result: {is_valid}")import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return dfimport pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                cleaned_df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value:.2f}")
    
    print(f"Cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if df.empty:
        print("Validation failed: DataFrame is empty.")
        return False
    
    if df.shape[0] < min_rows:
        print(f"Validation failed: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'], min_rows=3)
    print(f"\nData is valid: {is_valid}")
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_strategy='mean'):
    """
    Load and clean CSV data by handling missing values.
    
    Args:
        file_path (str): Path to the CSV file.
        fill_strategy (str): Strategy for filling missing values.
            Options: 'mean', 'median', 'zero', 'drop'.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    
    if df.empty:
        return df
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if fill_strategy == 'drop':
        df_cleaned = df.dropna(subset=numeric_columns)
    elif fill_strategy == 'zero':
        df_cleaned = df.copy()
        df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(0)
    elif fill_strategy == 'mean':
        df_cleaned = df.copy()
        for col in numeric_columns:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
    elif fill_strategy == 'median':
        df_cleaned = df.copy()
        for col in numeric_columns:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    else:
        raise ValueError(f"Unknown fill strategy: {fill_strategy}")
    
    return df_cleaned

def detect_outliers_iqr(df, column, multiplier=1.5):
    """
    Detect outliers using the Interquartile Range method.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        column (str): Column name to check for outliers.
        multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
        pandas.Series: Boolean mask indicating outliers.
    """
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column must be numeric: {column}")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return outliers

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        subset (list): Columns to consider for duplicates.
        keep (str): Which duplicates to keep.
    
    Returns:
        pandas.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep=keep)