
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_data(df, columns, method='minmax'):
    normalized_df = df.copy()
    for col in columns:
        if method == 'minmax':
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    return normalized_df

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    df_cleaned = remove_outliers_iqr(df, numeric_columns)
    df_normalized = normalize_data(df_cleaned, numeric_columns, method='zscore')
    return df_normalized

if __name__ == "__main__":
    cleaned_data = clean_dataset('sample_data.csv', ['age', 'income', 'score'])
    cleaned_data.to_csv('cleaned_sample_data.csv', index=False)
    print("Data cleaning completed. Cleaned data saved to 'cleaned_sample_data.csv'")