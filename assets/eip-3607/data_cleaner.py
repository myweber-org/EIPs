
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    return pd.read_csv(filepath)

def remove_outliers(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_data(df, numeric_columns):
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    return df.dropna()

if __name__ == "__main__":
    data = load_dataset("raw_data.csv")
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cleaned_data = clean_data(data, numeric_cols)
    cleaned_data.to_csv("cleaned_data.csv", index=False)
import re

def clean_string(input_string):
    """
    Cleans a string by:
    1. Stripping leading/trailing whitespace.
    2. Converting to lowercase.
    3. Replacing multiple spaces with a single space.
    4. Removing any non-alphanumeric characters except spaces.
    """
    if not isinstance(input_string, str):
        return input_string

    # Convert to lowercase and strip whitespace
    cleaned = input_string.lower().strip()

    # Remove any character that is not a letter, number, or space
    cleaned = re.sub(r'[^a-z0-9\s]', '', cleaned)

    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned