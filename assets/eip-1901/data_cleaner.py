
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Load, clean, and save CSV data.
    Removes duplicates, handles missing values, and standardizes formats.
    """
    try:
        df = pd.read_csv(input_path)
        
        original_rows = len(df)
        
        df = df.drop_duplicates()
        
        for column in df.select_dtypes(include=[np.number]).columns:
            df[column] = df[column].fillna(df[column].median())
        
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = df[column].fillna('Unknown')
            df[column] = df[column].str.strip().str.title()
        
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"cleaned_{input_file.name}"
        
        df.to_csv(output_path, index=False)
        
        cleaned_rows = len(df)
        removed_rows = original_rows - cleaned_rows
        
        print(f"Data cleaning completed successfully.")
        print(f"Original rows: {original_rows}")
        print(f"Cleaned rows: {cleaned_rows}")
        print(f"Removed rows: {removed_rows}")
        print(f"Output saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Validate the cleaned dataframe for common data quality issues.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'text_columns': len(df.select_dtypes(include=['object']).columns),
        'date_columns': len([col for col in df.columns if 'date' in col.lower()])
    }
    
    print("Data Validation Results:")
    for key, value in validation_results.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    return validation_results['missing_values'] == 0 and validation_results['duplicate_rows'] == 0

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', None],
        'age': [25, 30, 25, None, 35],
        'join_date': ['2023-01-15', '2023-02-20', '2023-01-15', '2023-03-10', '2023-04-05'],
        'salary': [50000, 60000, 50000, 55000, 52000]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_test_data.csv')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df)
        print(f"Data validation passed: {is_valid}")
        
        import os
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')
        if os.path.exists('cleaned_test_data.csv'):
            os.remove('cleaned_test_data.csv')
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: Optional dictionary to rename columns
        drop_duplicates: Whether to remove duplicate rows
        normalize_text: Whether to normalize text columns (strip, lowercase)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    return cleaned_df

def remove_special_characters(text, keep_pattern=r'[a-zA-Z0-9\s]'):
    """
    Remove special characters from text while keeping specified patterns.
    
    Args:
        text: Input string
        keep_pattern: Regex pattern of characters to keep
    
    Returns:
        Cleaned string
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    return re.sub(f'[^{keep_pattern}]', '', text)

def validate_email(email):
    """
    Validate email format using regex pattern.
    
    Args:
        email: Email string to validate
    
    Returns:
        Boolean indicating if email is valid
    """
    if pd.isna(email):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email).strip()))

def create_clean_data_pipeline(df, cleaning_steps=None):
    """
    Create a data cleaning pipeline with multiple steps.
    
    Args:
        df: Input DataFrame
        cleaning_steps: List of cleaning functions to apply
    
    Returns:
        Cleaned DataFrame after applying all steps
    """
    if cleaning_steps is None:
        cleaning_steps = [clean_dataframe]
    
    result = df.copy()
    for step in cleaning_steps:
        if callable(step):
            result = step(result)
    
    return result
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or np.array): The dataset.
    column (int): Index of the column to clean.
    
    Returns:
    np.array: Data with outliers removed.
    """
    data = np.array(data)
    col_data = data[:, column].astype(float)
    
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    cleaned_data = data[mask]
    
    return cleaned_data

def main():
    sample_data = [
        [1, 150.0, 'A'],
        [2, 200.0, 'B'],
        [3, 50.0, 'C'],
        [4, 300.0, 'D'],
        [5, 1000.0, 'E'],
        [6, 180.0, 'F']
    ]
    
    print("Original data:")
    for row in sample_data:
        print(row)
    
    cleaned = remove_outliers_iqr(sample_data, column=1)
    
    print("\nCleaned data (outliers removed from column 1):")
    for row in cleaned:
        print(row)

if __name__ == "__main__":
    main()import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        subset (list, optional): Column labels to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if dataframe.empty:
        return dataframe
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(dataframe, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    
    return dataframe

def validate_dataframe(dataframe, required_columns):
    """
    Validate that DataFrame contains required columns.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        required_columns (list): List of required column names
    
    Returns:
        bool: True if all required columns are present
    """
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    return True