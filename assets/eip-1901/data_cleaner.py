import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    """
    Clean CSV data by handling missing values and removing duplicates.
    """
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original shape: {df.shape}")
        
        df_cleaned = df.copy()
        
        df_cleaned = df_cleaned.drop_duplicates()
        
        for column in df_cleaned.columns:
            if df_cleaned[column].dtype in ['int64', 'float64']:
                df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
            elif df_cleaned[column].dtype == 'object':
                df_cleaned[column].fillna(df_cleaned[column].mode()[0], inplace=True)
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Cleaned shape: {df_cleaned.shape}")
        print(f"Data saved to: {output_file}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    success = clean_csv_data(input_csv, output_csv)
    
    if success:
        print("Data cleaning completed successfully.")
    else:
        print("Data cleaning failed.")