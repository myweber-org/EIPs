
import pandas as pd
import numpy as np
from datetime import datetime

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    
    df.drop_duplicates(inplace=True)
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['amount'] = df['amount'].fillna(0)
    
    df['category'] = df['category'].str.lower().str.strip()
    df['category'] = df['category'].replace({'': 'uncategorized'})
    
    df = df.dropna(subset=['date', 'description'])
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    print(f"Original rows: {len(pd.read_csv(input_file))}, Cleaned rows: {len(df)}")
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')