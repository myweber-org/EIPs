
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