
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    else:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
        method (str): Outlier detection method ('iqr' or 'zscore')
        threshold (float): Threshold for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        filtered_df = df[z_scores < threshold]
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return filtered_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, np.nan, 7, 8, 9, 10],
        'B': [10, 20, 20, 40, 50, 60, 70, 80, 90, 100],
        'C': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and clean CSV data by handling missing values and converting data types.
    
    Parameters:
    input_path: Path to the input CSV file
    output_path: Optional path to save cleaned data
    
    Returns:
    Cleaned pandas DataFrame
    """
    try:
        df = pd.read_csv(input_path)
        
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        # Handle missing values
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:
                # Fill numeric columns with median
                df[column] = df[column].fillna(df[column].median())
            elif df[column].dtype == 'object':
                # Fill categorical columns with mode
                df[column] = df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 'Unknown')
        
        # Convert date columns if present
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                print(f"Could not convert column {col} to datetime")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        
        print(f"Removed {removed_duplicates} duplicate rows")
        print(f"Cleaned data shape: {df.shape}")
        print(f"Missing values after cleaning:\n{df.isnull().sum()}")
        
        # Save cleaned data if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        raise
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        raise

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame meets basic quality criteria.
    
    Parameters:
    df: DataFrame to validate
    
    Returns:
    Boolean indicating if data passes validation
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    # Check for remaining missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Validation warning: {missing_values} missing values remain")
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            print(f"Validation warning: Infinite values found in column {col}")
    
    return True

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_data = clean_csv_data(input_file, output_file)
        
        if validate_dataframe(cleaned_data):
            print("Data cleaning completed successfully")
            print(f"Data types:\n{cleaned_data.dtypes}")
        else:
            print("Data cleaning completed with warnings")
            
    except Exception as e:
        print(f"Data cleaning failed: {e}")