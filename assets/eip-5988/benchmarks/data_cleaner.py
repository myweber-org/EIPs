import pandas as pd
import numpy as np

def clean_csv_data(input_path, output_path):
    """
    Load CSV data, handle missing values, convert data types,
    and save cleaned version.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        
        # Convert date columns if present
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        print(f"Data cleaned successfully. Saved to {output_path}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file {input_path} not found")
        return False
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    clean_csv_data('raw_data.csv', 'cleaned_data.csv')