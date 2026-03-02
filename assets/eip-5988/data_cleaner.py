import numpy as np
import pandas as pd

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
    if max_val - min_val == 0:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    if std_val == 0:
        return df[column]
    return (df[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    return cleaned_df

def process_features(df, numeric_columns, method='standardize'):
    processed_df = df.copy()
    for col in numeric_columns:
        if col in processed_df.columns:
            if method == 'normalize':
                processed_df[col] = normalize_minmax(processed_df, col)
            elif method == 'standardize':
                processed_df[col] = standardize_zscore(processed_df, col)
    return processed_df

if __name__ == "__main__":
    sample_data = {
        'feature1': [1, 2, 3, 4, 100],
        'feature2': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'B', 'A']
    }
    df = pd.DataFrame(sample_data)
    numeric_cols = ['feature1', 'feature2']
    
    cleaned = clean_dataset(df, numeric_cols)
    print("Cleaned dataset:")
    print(cleaned)
    
    standardized = process_features(cleaned, numeric_cols, 'standardize')
    print("\nStandardized features:")
    print(standardized)