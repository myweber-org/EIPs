
import pandas as pd
import numpy as np
from pathlib import Path

def clean_dataset(input_path, output_path=None):
    """
    Load a CSV dataset, remove duplicate rows, standardize column names,
    and fill missing numeric values with column median.
    """
    df = pd.read_csv(input_path)
    
    original_shape = df.shape
    print(f"Original dataset shape: {original_shape}")
    
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    df = df.drop_duplicates()
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled missing values in '{col}' with median: {median_val}")
    
    cleaned_shape = df.shape
    print(f"Cleaned dataset shape: {cleaned_shape}")
    print(f"Removed {original_shape[0] - cleaned_shape[0]} duplicate rows")
    
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_cleaned{input_file.suffix}"
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        clean_dataset(input_file, output_file)
    else:
        print("Usage: python data_cleaner.py <input_file> [output_file]")