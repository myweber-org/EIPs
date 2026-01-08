
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values with column mean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                mean_value = cleaned_df[column].mean()
                cleaned_df[column] = cleaned_df[column].fillna(mean_value)
                print(f"Filled missing values in '{column}' with mean: {mean_value:.2f}")
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the structure and content of a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
    
    Returns:
        dict: Dictionary containing validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_cols
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation results:")
    print(validate_data(df))
    
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nCleaned validation results:")
    print(validate_data(cleaned))import csv
import re

def clean_numeric_string(value):
    """Remove non-numeric characters from a string and convert to integer."""
    if not value:
        return None
    cleaned = re.sub(r'[^\d.-]', '', str(value))
    try:
        return int(cleaned) if '.' not in cleaned else float(cleaned)
    except ValueError:
        return None

def normalize_column_names(headers):
    """Normalize column names to lowercase with underscores."""
    return [re.sub(r'\s+', '_', header.strip().lower()) for header in headers]

def read_and_clean_csv(file_path, delimiter=','):
    """Read a CSV file and clean its data."""
    cleaned_data = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        headers = next(reader)
        normalized_headers = normalize_column_names(headers)
        
        for row in reader:
            cleaned_row = {}
            for header, value in zip(normalized_headers, row):
                if any(keyword in header for keyword in ['id', 'code', 'number']):
                    cleaned_row[header] = clean_numeric_string(value)
                else:
                    cleaned_row[header] = value.strip() if value else None
            cleaned_data.append(cleaned_row)
    
    return cleaned_data

def write_cleaned_csv(data, output_path, delimiter=','):
    """Write cleaned data to a new CSV file."""
    if not data:
        return
    
    headers = list(data[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(data)

def remove_duplicates(data, key_columns):
    """Remove duplicate rows based on specified key columns."""
    seen = set()
    unique_data = []
    
    for row in data:
        key = tuple(row[col] for col in key_columns if col in row)
        if key not in seen:
            seen.add(key)
            unique_data.append(row)
    
    return unique_data