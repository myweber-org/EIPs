import pandas as pd
import numpy as np
from datetime import datetime

def clean_csv_data(input_path, output_path):
    """
    Load a CSV file, perform cleaning operations, and save the cleaned data.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Original data shape: {df.shape}")
        
        # Remove duplicate rows
        df.drop_duplicates(inplace=True)
        print(f"After removing duplicates: {df.shape}")
        
        # Handle missing values for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Handle missing values for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna('Unknown', inplace=True)
        
        # Convert date columns if present
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        # Remove columns with too many missing values
        threshold = 0.7
        cols_to_drop = [col for col in df.columns if df[col].isnull().sum() / len(df) > threshold]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            print(f"Dropped columns with >{threshold*100}% missing values: {cols_to_drop}")
        
        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        print(f"Final data shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    
    if cleaned_df is not None:
        print("Data cleaning completed successfully.")
        print("\nSample of cleaned data:")
        print(cleaned_df.head())