import pandas as pd
import numpy as np

def load_data(filepath):
    """Load data from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers(df, column, threshold=3):
    """Remove outliers using z-score method."""
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]

def normalize_column(df, column):
    """Normalize column to range [0,1]."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_data(input_file, output_file, target_column):
    """Main data cleaning pipeline."""
    df = load_data(input_file)
    print(f"Original data shape: {df.shape}")
    
    df = remove_outliers(df, target_column)
    print(f"After outlier removal: {df.shape}")
    
    df = normalize_column(df, target_column)
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to: {output_file}")
    return df

if __name__ == "__main__":
    cleaned_df = clean_data("raw_data.csv", "cleaned_data.csv", "value")