import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    """
    Clean CSV data by handling missing values and converting data types.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
    """
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        print(f"Missing values before cleaning:\n{df.isnull().sum()}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        for col in categorical_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except (ValueError, TypeError):
                    pass
        
        df = df.drop_duplicates()
        
        print(f"Cleaned data shape: {df.shape}")
        print(f"Missing values after cleaning:\n{df.isnull().sum()}")
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df, required_columns=None):
    """
    Validate cleaned data for required columns and basic integrity.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None.")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Validation failed: Missing required columns: {missing_columns}")
            return False
    
    if df.isnull().any().any():
        print("Validation failed: DataFrame contains null values.")
        return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_csv, output_csv)
    
    if cleaned_df is not None:
        required_cols = ['id', 'timestamp', 'value']
        validation_result = validate_data(cleaned_df, required_cols)
        
        if validation_result:
            print("Data cleaning and validation completed successfully.")
        else:
            print("Data cleaning completed but validation failed.")