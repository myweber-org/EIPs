
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    df = pd.read_csv(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    df = df.drop_duplicates()
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    cleaned_df = clean_csv_data(input_csv, output_csv)