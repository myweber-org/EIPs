
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
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
        if col in normalized_df.columns:
            if method == 'minmax':
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val > 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    return normalized_df

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    if outlier_method == 'iqr':
        df_cleaned = remove_outliers_iqr(df, numeric_columns)
    else:
        df_cleaned = df.copy()
    
    df_normalized = normalize_data(df_cleaned, numeric_columns, method=normalize_method)
    return df_normalized

if __name__ == "__main__":
    sample_data = {
        'feature1': [10, 12, 12, 14, 15, 100, 13, 11, 9, 16],
        'feature2': [20, 22, 21, 25, 23, 200, 24, 19, 18, 26],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    numeric_cols = ['feature1', 'feature2']
    
    cleaned_df = clean_dataset(df, numeric_cols)
    print("Original dataset:")
    print(df)
    print("\nCleaned and normalized dataset:")
    print(cleaned_df)