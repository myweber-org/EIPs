
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
        print("Filled missing values with mode")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, None, 15.2, 20.1, None, 30.7],
        'category': ['A', 'B', 'B', 'A', 'C', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataframe(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, required_columns=['id', 'value'], min_rows=3)
    print(f"\nDataFrame validation: {'PASSED' if is_valid else 'FAILED'}")import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean'):
    """
    Load and clean a CSV file by handling missing values.
    
    Args:
        filepath (str): Path to the CSV file.
        fill_strategy (str): Strategy for filling missing values.
            Options: 'mean', 'median', 'zero', 'drop'.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    print(f"Original shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    if fill_strategy == 'drop':
        df_cleaned = df.dropna()
    elif fill_strategy == 'zero':
        df_cleaned = df.fillna(0)
    elif fill_strategy == 'mean':
        df_cleaned = df.fillna(df.mean(numeric_only=True))
    elif fill_strategy == 'median':
        df_cleaned = df.fillna(df.median(numeric_only=True))
    else:
        raise ValueError(f"Unknown fill strategy: {fill_strategy}")
    
    print(f"Cleaned shape: {df_cleaned.shape}")
    print(f"Remaining missing values: {df_cleaned.isnull().sum().sum()}")
    
    return df_cleaned

def detect_outliers_iqr(df, column):
    """
    Detect outliers in a column using the IQR method.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        column (str): Column name to analyze.
    
    Returns:
        pandas.DataFrame: Rows identified as outliers.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not np.issubdtype(df[column].dtype, np.number):
        raise TypeError(f"Column '{column}' must be numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return outliers

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, np.nan, 30, 40, 50, 60],
        'C': ['x', 'y', 'z', 'x', 'y', 'z']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned = clean_csv_data('sample_data.csv', fill_strategy='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    outliers = detect_outliers_iqr(cleaned, 'A')
    print(f"\nOutliers in column 'A': {len(outliers)} rows")
    print(outliers)