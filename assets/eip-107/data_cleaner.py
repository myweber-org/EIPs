
import pandas as pd
import numpy as np
from datetime import datetime

def clean_csv_data(input_path, output_path):
    """
    Clean CSV data by handling missing values and converting data types.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Fill missing numeric values with column mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()))
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Convert date columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except (ValueError, TypeError):
                    continue
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        print(f"Data cleaning completed. Cleaned data saved to {output_path}")
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df):
    """
    Validate cleaned data for common issues.
    """
    if df is None:
        return False
    
    checks = {
        'has_nulls': df.isnull().sum().sum() == 0,
        'has_duplicates': len(df) == len(df.drop_duplicates()),
        'has_negative_values': (df.select_dtypes(include=[np.number]) < 0).sum().sum() == 0
    }
    
    return all(checks.values())

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    
    if cleaned_df is not None:
        is_valid = validate_data(cleaned_df)
        print(f"Data validation: {'PASSED' if is_valid else 'FAILED'}")