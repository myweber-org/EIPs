
import re

def clean_string(text):
    """
    Clean a string by removing extra whitespace and converting to lowercase.
    
    Args:
        text (str): The input string to clean.
    
    Returns:
        str: The cleaned string.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def normalize_names(name_list):
    """
    Normalize a list of names by cleaning each string.
    
    Args:
        name_list (list): List of name strings.
    
    Returns:
        list: List of cleaned names.
    """
    if not isinstance(name_list, list):
        raise TypeError("Input must be a list")
    
    return [clean_string(name) for name in name_list]
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
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
        'count': len(df[column]),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original summary statistics:")
    print(calculate_summary_stats(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary statistics:")
    print(calculate_summary_stats(cleaned_df, 'value'))
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        columns_to_check: list of columns to check for duplicates (default: all columns)
        fill_missing: boolean indicating whether to fill missing values (default: True)
    
    Returns:
        Cleaned pandas DataFrame
    """
    # Create a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if fill_missing:
        # Fill numeric columns with median
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        
        # Fill categorical columns with mode
        categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if cleaned_df[col].isnull().any():
                mode_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown'
                cleaned_df[col] = cleaned_df[col].fillna(mode_value)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a pandas DataFrame for required columns and data quality.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names (default: None)
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'empty_rows': 0,
        'total_rows': len(df),
        'total_columns': len(df.columns)
    }
    
    # Check for required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['missing_columns'] = missing_cols
            validation_results['is_valid'] = False
    
    # Check for empty rows
    empty_rows = df.isnull().all(axis=1).sum()
    validation_results['empty_rows'] = empty_rows
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
        'age': [25, 30, 30, None, 28, 35],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, columns_to_check=['id', 'name'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    # Validate the cleaned dataset
    validation = validate_dataset(cleaned_df, required_columns=['id', 'name', 'age'])
    print("Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")
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
    
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values.")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values.")
        elif fill_missing == 'mean':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
            print("Filled missing numeric values with column means.")
        elif fill_missing == 'median':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
            print("Filled missing numeric values with column medians.")
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown'
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val)
            print("Filled missing categorical values with column modes.")
        else:
            print(f"Unknown fill method: {fill_missing}. No filling performed.")
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for basic integrity checks.
    
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
    
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("DataFrame validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None, 6],
        'B': [10, 20, 20, None, 50, 60],
        'C': ['X', 'Y', 'Y', 'Z', None, 'X']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, required_columns=['A', 'B'], min_rows=3)
    print(f"\nDataFrame is valid: {is_valid}")