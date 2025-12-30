
def remove_duplicates(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (bool): Whether to fill missing values
    
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
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in ['int64', 'float64']:
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
            elif cleaned_df[column].dtype == 'object':
                cleaned_df[column] = cleaned_df[column].fillna('Unknown')
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate DataFrame for common data quality issues.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    
    Returns:
    dict: Dictionary containing validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    return validation_results

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Parameters:
    df (pd.DataFrame): DataFrame to save
    output_path (str): Path to save the file
    format (str): File format ('csv' or 'parquet')
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Data saved to {output_path} in {format} format")

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_dataframe(df))
    
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Validation Results:")
    print(validate_dataframe(cleaned_df))import csv
import os

def clean_csv(input_path, output_path, remove_duplicates=True, strip_whitespace=True):
    """
    Clean a CSV file by removing duplicates and stripping whitespace.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    seen_rows = set()
    cleaned_rows = []
    
    with open(input_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        cleaned_rows.append(header)
        
        for row in reader:
            if strip_whitespace:
                row = [cell.strip() if isinstance(cell, str) else cell for cell in row]
            
            row_tuple = tuple(row)
            
            if remove_duplicates:
                if row_tuple in seen_rows:
                    continue
                seen_rows.add(row_tuple)
            
            cleaned_rows.append(row)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_rows)
    
    return len(cleaned_rows) - 1

def validate_csv(file_path, required_columns=None):
    """
    Validate CSV structure and required columns.
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)
            
            if required_columns:
                missing_columns = [col for col in required_columns if col not in header]
                if missing_columns:
                    return False, f"Missing required columns: {missing_columns}"
            
            row_count = 0
            for row in reader:
                row_count += 1
                if len(row) != len(header):
                    return False, f"Row {row_count} has {len(row)} columns, expected {len(header)}"
            
            return True, f"CSV validation passed: {row_count} rows processed"
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def merge_csv_files(file_paths, output_path, deduplicate=True):
    """
    Merge multiple CSV files into one.
    """
    if not file_paths:
        raise ValueError("No input files provided")
    
    all_rows = []
    header = None
    seen_rows = set()
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping: {file_path}")
            continue
        
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            current_header = next(reader)
            
            if header is None:
                header = current_header
            elif header != current_header:
                print(f"Warning: Header mismatch in {file_path}, skipping file")
                continue
            
            for row in reader:
                row_tuple = tuple(row)
                if deduplicate and row_tuple in seen_rows:
                    continue
                
                if deduplicate:
                    seen_rows.add(row_tuple)
                all_rows.append(row)
    
    if header is None:
        raise ValueError("No valid CSV files found")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(all_rows)
    
    return len(all_rows)

def sample_csv(input_path, output_path, sample_size=100, random_seed=42):
    """
    Create a random sample from a CSV file.
    """
    import random
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        all_rows = list(reader)
    
    if sample_size >= len(all_rows):
        sampled_rows = all_rows
    else:
        random.seed(random_seed)
        sampled_rows = random.sample(all_rows, sample_size)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(sampled_rows)
    
    return len(sampled_rows)

if __name__ == "__main__":
    # Example usage
    try:
        rows_cleaned = clean_csv("input.csv", "cleaned.csv")
        print(f"Cleaned {rows_cleaned} rows")
        
        is_valid, message = validate_csv("cleaned.csv", ["id", "name"])
        print(f"Validation: {is_valid} - {message}")
        
    except Exception as e:
        print(f"Error: {e}")