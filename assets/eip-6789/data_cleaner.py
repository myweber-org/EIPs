import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    """
    Load a CSV file and perform basic data cleaning operations.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Remove duplicate rows
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_count - len(df)
    print(f"Removed {duplicates_removed} duplicate rows.")

    # Handle missing values for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"Filled missing values in column '{col}' with median.")

    # Remove outliers using Z-score for numeric columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    outlier_mask = (z_scores < 3).all(axis=1)
    outliers_removed = len(df) - outlier_mask.sum()
    df = df[outlier_mask]
    print(f"Removed {outliers_removed} outliers based on Z-score (|Z| > 3).")

    # Normalize numeric columns to range [0, 1]
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
            print(f"Normalized column '{col}' to range [0, 1].")
        else:
            print(f"Column '{col}' has constant values, skipping normalization.")

    print(f"Final data shape: {df.shape}")
    return df

def save_cleaned_data(df, output_filepath):
    """
    Save the cleaned DataFrame to a CSV file.
    """
    try:
        df.to_csv(output_filepath, index=False)
        print(f"Cleaned data saved to {output_filepath}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"

    cleaned_df = load_and_clean_data(input_file)
    if cleaned_df is not None:
        save_cleaned_data(cleaned_df, output_file)