
import pandas as pd
from datetime import datetime

def clean_dataframe(df, date_column='date', id_column='id'):
    """
    Clean a DataFrame by removing duplicate IDs and standardizing date formats.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        date_column (str): Name of the date column to standardize
        id_column (str): Name of the ID column for duplicate removal
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Remove duplicate entries based on ID column
    df_clean = df.drop_duplicates(subset=[id_column], keep='first')
    
    # Standardize date format to YYYY-MM-DD
    if date_column in df_clean.columns:
        df_clean[date_column] = pd.to_datetime(df_clean[date_column], errors='coerce')
        df_clean[date_column] = df_clean[date_column].dt.strftime('%Y-%m-%d')
    
    # Remove rows with invalid dates after conversion
    df_clean = df_clean.dropna(subset=[date_column])
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def validate_data(df, required_columns=None):
    """
    Validate that DataFrame contains required columns and has data.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def process_file(input_path, output_path, date_column='date', id_column='id'):
    """
    Read, clean, and save data from CSV file.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save cleaned CSV file
        date_column (str): Name of date column
        id_column (str): Name of ID column
    """
    try:
        # Read input file
        df = pd.read_csv(input_path)
        
        # Validate data
        if not validate_data(df, required_columns=[date_column, id_column]):
            return
        
        # Clean data
        df_clean = clean_dataframe(df, date_column, id_column)
        
        # Save cleaned data
        df_clean.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        print(f"Original rows: {len(df)}, Cleaned rows: {len(df_clean)}")
        
    except FileNotFoundError:
        print(f"Input file not found: {input_path}")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    process_file(input_file, output_file)
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                print(f"Filled missing values in column '{col}' with median")
        
        categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if cleaned_df[col].isnull().any():
                cleaned_df[col] = cleaned_df[col].fillna('Unknown')
                print(f"Filled missing values in column '{col}' with 'Unknown'")
    
    return cleaned_df

def validate_dataset(df):
    """
    Validate the cleaned dataset for common data quality issues.
    """
    validation_results = {}
    
    validation_results['total_rows'] = len(df)
    validation_results['total_columns'] = len(df.columns)
    validation_results['missing_values'] = df.isnull().sum().sum()
    validation_results['duplicate_rows'] = df.duplicated().sum()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    validation_results['numeric_columns'] = len(numeric_cols)
    
    for col in numeric_cols:
        validation_results[f'{col}_min'] = df[col].min()
        validation_results[f'{col}_max'] = df[col].max()
        validation_results[f'{col}_mean'] = df[col].mean()
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve', 'Eve'],
        'age': [25, 30, 30, None, 35, 40, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, 95.5, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned dataset:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    validation = validate_dataset(cleaned_df)
    print("Validation results:")
    for key, value in validation.items():
        print(f"{key}: {value}")