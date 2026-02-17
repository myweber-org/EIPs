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
        return df[column].apply(lambda x: 0.0)
    return (df[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df

def validate_dataframe(df):
    required_checks = [
        (lambda x: not x.empty, "DataFrame is empty"),
        (lambda x: x.isnull().sum().sum() == 0, "DataFrame contains null values"),
        (lambda x: all(x.dtypes != object), "DataFrame contains non-numeric columns")
    ]
    for check_func, error_msg in required_checks:
        if not check_func(df):
            raise ValueError(f"Validation failed: {error_msg}")
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(2.0, 1000),
        'feature_c': np.random.uniform(0, 1, 1000)
    })
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    processed_data = clean_dataset(sample_data, numeric_cols)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {processed_data.shape}")
    print("Data cleaning completed successfully.")