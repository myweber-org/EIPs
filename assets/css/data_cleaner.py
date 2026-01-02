
import csv
import re
from typing import List, Optional

def clean_csv_data(input_file: str, output_file: str, columns_to_clean: Optional[List[str]] = None) -> None:
    """
    Clean a CSV file by removing extra whitespace and standardizing text.
    
    Args:
        input_file: Path to the input CSV file.
        output_file: Path to save the cleaned CSV file.
        columns_to_clean: List of column names to apply cleaning. If None, clean all columns.
    """
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        if fieldnames is None:
            raise ValueError("CSV file has no headers")
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        columns_to_process = columns_to_clean if columns_to_clean else fieldnames
        
        for row in reader:
            cleaned_row = {}
            for field in fieldnames:
                value = row.get(field, '')
                if field in columns_to_process and isinstance(value, str):
                    # Remove extra whitespace
                    value = re.sub(r'\s+', ' ', value.strip())
                    # Standardize capitalization for certain patterns
                    value = value.lower() if value.isupper() else value
                cleaned_row[field] = value
            writer.writerow(cleaned_row)

def validate_email_in_column(csv_file: str, email_column: str) -> List[dict]:
    """
    Validate email addresses in a specific column of a CSV file.
    
    Args:
        csv_file: Path to the CSV file.
        email_column: Name of the column containing email addresses.
    
    Returns:
        List of dictionaries with validation results.
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    results = []
    
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for i, row in enumerate(reader, start=2):  # start=2 for line numbers (header is line 1)
            email = row.get(email_column, '')
            is_valid = bool(re.match(email_pattern, email))
            
            results.append({
                'line': i,
                'email': email,
                'is_valid': is_valid,
                'original_row': row
            })
    
    return results

def remove_duplicates(csv_file: str, unique_columns: List[str], output_file: str) -> None:
    """
    Remove duplicate rows based on specified columns.
    
    Args:
        csv_file: Path to the input CSV file.
        unique_columns: List of column names to identify duplicates.
        output_file: Path to save the deduplicated CSV file.
    """
    seen = set()
    unique_rows = []
    
    with open(csv_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        if fieldnames is None:
            raise ValueError("CSV file has no headers")
        
        for row in reader:
            # Create a tuple of values from the specified unique columns
            key_tuple = tuple(str(row.get(col, '')).strip().lower() for col in unique_columns)
            
            if key_tuple not in seen:
                seen.add(key_tuple)
                unique_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)

if __name__ == "__main__":
    # Example usage
    clean_csv_data("raw_data.csv", "cleaned_data.csv")
    
    validation_results = validate_email_in_column("cleaned_data.csv", "email")
    invalid_emails = [r for r in validation_results if not r['is_valid']]
    print(f"Found {len(invalid_emails)} invalid email addresses")
    
    remove_duplicates("cleaned_data.csv", ["email", "name"], "deduplicated_data.csv")
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

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name
    
    Returns:
        dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list): List of numeric column names. If None, uses all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal statistics for column 'A':")
    print(calculate_statistics(df, 'A'))
    
    cleaned_df = clean_dataset(df, ['A', 'B'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned statistics for column 'A':")
    print(calculate_statistics(cleaned_df, 'A'))