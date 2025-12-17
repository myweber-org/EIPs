
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_column(df, column):
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

def clean_data(df, numeric_columns):
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    return df.dropna()

if __name__ == "__main__":
    data = load_data("raw_data.csv")
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cleaned_data = clean_data(data, numeric_cols)
    cleaned_data.to_csv("cleaned_data.csv", index=False)
    print(f"Data cleaned. Original: {len(data)} rows, Cleaned: {len(cleaned_data)} rows")