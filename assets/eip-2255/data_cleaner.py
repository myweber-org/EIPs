
import pandas as pd
import numpy as np
from datetime import datetime

def clean_csv_data(input_file, output_file):
    """
    Clean CSV data by handling missing values and converting data types.
    """
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Convert date columns if present
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        
        print(f"Cleaned shape: {df.shape}")
        print(f"Removed duplicates: {removed_duplicates}")
        print(f"Cleaned data saved to: {output_file}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None

def validate_data_types(df):
    """
    Validate and report data types in the dataframe.
    """
    if df is not None:
        print("\nData types after cleaning:")
        for col in df.columns:
            dtype = df[col].dtype
            unique_count = df[col].nunique()
            print(f"{col}: {dtype} | Unique values: {unique_count}")
        
        return True
    return False

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_csv, output_csv)
    
    if cleaned_df is not None:
        validate_data_types(cleaned_df)import csv
import re

def clean_csv(input_file, output_file, remove_duplicates=True, strip_whitespace=True):
    """
    Clean a CSV file by removing duplicates and stripping whitespace.
    """
    cleaned_rows = []
    seen_rows = set()
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
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
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_rows)
    
    return len(cleaned_rows) - 1

def validate_email_column(input_file, email_column_index):
    """
    Validate email addresses in a specific column of a CSV file.
    """
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    invalid_emails = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        
        for row_number, row in enumerate(reader, start=2):
            if email_column_index < len(row):
                email = row[email_column_index]
                if not email_pattern.match(email):
                    invalid_emails.append((row_number, email))
    
    return invalid_emails

def split_csv_by_column(input_file, output_prefix, split_column_index):
    """
    Split a CSV file into multiple files based on unique values in a column.
    """
    file_handles = {}
    writers = {}
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        
        for row in reader:
            if split_column_index < len(row):
                key = row[split_column_index]
                
                if key not in file_handles:
                    filename = f"{output_prefix}_{key}.csv"
                    file_handle = open(filename, 'w', newline='', encoding='utf-8')
                    writer = csv.writer(file_handle)
                    writer.writerow(header)
                    
                    file_handles[key] = file_handle
                    writers[key] = writer
                
                writers[key].writerow(row)
    
    for handle in file_handles.values():
        handle.close()
    
    return list(file_handles.keys())
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric columns by removing outliers from specified columns or all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list, optional): List of column names to clean. If None, cleans all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not clean column '{col}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 21),
        'value': [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                  21, 22, 23, 24, 25, 100, 120, 150, 200, 5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data shape:", df.shape)
    print("Original data:")
    print(df)
    
    cleaned_df = clean_numeric_data(df, columns=['value'])
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Cleaned data:")
    print(cleaned_df)