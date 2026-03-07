
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
        missing_strategy (str): Strategy for handling missing values
                               ('mean', 'median', 'drop', 'zero')
    """
    try:
        df = pd.read_csv(input_file)
        
        original_rows = len(df)
        original_columns = len(df.columns)
        
        df = df.drop_duplicates()
        
        if missing_strategy == 'drop':
            df = df.dropna()
        elif missing_strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
        elif missing_strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
        elif missing_strategy == 'zero':
            df = df.fillna(0)
        
        df.to_csv(output_file, index=False)
        
        cleaned_rows = len(df)
        cleaned_columns = len(df.columns)
        
        print(f"Data cleaning completed:")
        print(f"Original data: {original_rows} rows, {original_columns} columns")
        print(f"Cleaned data: {cleaned_rows} rows, {cleaned_columns} columns")
        print(f"Duplicates removed: {original_rows - cleaned_rows}")
        print(f"Output saved to: {output_file}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("Validation failed: Dataframe is empty or None")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Validation failed: Missing required columns: {missing_columns}")
            return False
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("Warning: No numeric columns found in dataframe")
    
    return True

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_data = clean_csv_data(input_csv, output_csv, missing_strategy='mean')
    
    if cleaned_data is not None:
        is_valid = validate_dataframe(cleaned_data)
        if is_valid:
            print("Data validation passed")
        else:
            print("Data validation failed")