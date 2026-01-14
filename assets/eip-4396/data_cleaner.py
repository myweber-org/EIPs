
import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        file_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned output CSV. 
                                     If None, returns DataFrame
        missing_strategy (str): Strategy for handling missing values.
                               Options: 'mean', 'median', 'drop', 'zero'
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None
    """
    try:
        df = pd.read_csv(file_path)
        
        original_shape = df.shape
        print(f"Original data shape: {original_shape}")
        
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mean(), inplace=True)
        elif missing_strategy == 'median':
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
        elif missing_strategy == 'zero':
            df.fillna(0, inplace=True)
        elif missing_strategy == 'drop':
            df.dropna(inplace=True)
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} missing values remain")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            print(f"Cleaned data shape: {df.shape}")
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    if df.isnull().any().any():
        print("Warning: DataFrame contains missing values")
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, np.nan, 15.2, np.nan, 20.1, 20.1],
        'category': ['A', 'B', 'A', 'B', 'A', 'A']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', missing_strategy='mean')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
        print(f"Data validation result: {is_valid}")
        print("\nCleaned data preview:")
        print(cleaned_df.head())import csv
import sys
from pathlib import Path

def clean_csv(input_path, output_path=None):
    """
    Clean a CSV file by removing rows with missing values
    and standardizing column names.
    """
    if output_path is None:
        input_stem = Path(input_path).stem
        output_path = f"{input_stem}_cleaned.csv"
    
    cleaned_rows = []
    
    try:
        with open(input_path, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            headers = next(reader)
            
            # Clean headers
            cleaned_headers = [
                header.strip().lower().replace(' ', '_')
                for header in headers
            ]
            cleaned_rows.append(cleaned_headers)
            
            # Process rows
            for row in reader:
                # Skip rows with missing values
                if any(cell.strip() == '' for cell in row):
                    continue
                
                # Clean cell values
                cleaned_row = [cell.strip() for cell in row]
                cleaned_rows.append(cleaned_row)
        
        # Write cleaned data
        with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(cleaned_rows)
        
        print(f"Cleaned data saved to: {output_path}")
        print(f"Original rows: {len(cleaned_rows) - 1}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def validate_csv(file_path):
    """
    Validate CSV file structure and content.
    """
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader)
            
            row_count = 1
            for row in reader:
                row_count += 1
                if len(row) != len(headers):
                    print(f"Warning: Row {row_count} has {len(row)} columns, expected {len(headers)}")
            
            print(f"CSV validation complete.")
            print(f"Headers: {len(headers)}")
            print(f"Data rows: {row_count - 1}")
            
    except Exception as e:
        print(f"Validation error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_csv> [output_csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Processing: {input_file}")
    clean_csv(input_file, output_file)
    validate_csv(output_file or f"{Path(input_file).stem}_cleaned.csv")