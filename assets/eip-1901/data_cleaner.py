
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Load, clean, and save CSV data.
    Removes duplicates, handles missing values, and standardizes formats.
    """
    try:
        df = pd.read_csv(input_path)
        
        original_rows = len(df)
        
        df = df.drop_duplicates()
        
        for column in df.select_dtypes(include=[np.number]).columns:
            df[column] = df[column].fillna(df[column].median())
        
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = df[column].fillna('Unknown')
            df[column] = df[column].str.strip().str.title()
        
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"cleaned_{input_file.name}"
        
        df.to_csv(output_path, index=False)
        
        cleaned_rows = len(df)
        removed_rows = original_rows - cleaned_rows
        
        print(f"Data cleaning completed successfully.")
        print(f"Original rows: {original_rows}")
        print(f"Cleaned rows: {cleaned_rows}")
        print(f"Removed rows: {removed_rows}")
        print(f"Output saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Validate the cleaned dataframe for common data quality issues.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'text_columns': len(df.select_dtypes(include=['object']).columns),
        'date_columns': len([col for col in df.columns if 'date' in col.lower()])
    }
    
    print("Data Validation Results:")
    for key, value in validation_results.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    return validation_results['missing_values'] == 0 and validation_results['duplicate_rows'] == 0

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', None],
        'age': [25, 30, 25, None, 35],
        'join_date': ['2023-01-15', '2023-02-20', '2023-01-15', '2023-03-10', '2023-04-05'],
        'salary': [50000, 60000, 50000, 55000, 52000]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_test_data.csv')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df)
        print(f"Data validation passed: {is_valid}")
        
        import os
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')
        if os.path.exists('cleaned_test_data.csv'):
            os.remove('cleaned_test_data.csv')