
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    """
    Clean CSV data by handling missing values and removing duplicates.
    """
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        
        df_cleaned = df.copy()
        
        df_cleaned = df_cleaned.drop_duplicates()
        
        for column in df_cleaned.columns:
            if df_cleaned[column].dtype in ['float64', 'int64']:
                df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
            elif df_cleaned[column].dtype == 'object':
                df_cleaned[column].fillna(df_cleaned[column].mode()[0], inplace=True)
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Cleaned data shape: {df_cleaned.shape}")
        print(f"Data saved to: {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
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
    
    validation_results = {
        'has_missing_values': df.isnull().sum().sum() == 0,
        'has_duplicates': len(df) == len(df.drop_duplicates()),
        'data_types_consistent': True
    }
    
    for column in df.columns:
        if df[column].dtype not in ['float64', 'int64', 'object', 'bool', 'datetime64[ns]']:
            validation_results['data_types_consistent'] = False
            break
    
    return validation_results

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_data = clean_csv_data(input_csv, output_csv)
    
    if cleaned_data is not None:
        validation = validate_data(cleaned_data)
        print("Data validation results:")
        for key, value in validation.items():
            print(f"  {key}: {value}")