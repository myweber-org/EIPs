import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def handle_missing_mean(df, column):
    mean_val = df[column].mean()
    df[column].fillna(mean_val, inplace=True)
    return df

def process_dataset(df, numeric_columns):
    processed_df = df.copy()
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df = remove_outliers_iqr(processed_df, col)
            processed_df = normalize_minmax(processed_df, col)
            processed_df = standardize_zscore(processed_df, col)
            processed_df = handle_missing_mean(processed_df, col)
    return processed_df

if __name__ == "__main__":
    sample_data = {'A': [1, 2, 3, 4, 5, 100],
                   'B': [10, 20, 30, 40, 50, 200],
                   'C': [0.1, 0.2, 0.3, 0.4, 0.5, 2.0]}
    df = pd.DataFrame(sample_data)
    result = process_dataset(df, ['A', 'B', 'C'])
    print(result.head())