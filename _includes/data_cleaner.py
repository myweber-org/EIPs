
import pandas as pd
import re

def clean_text_column(series):
    """
    Standardize text: lowercase, strip whitespace, remove extra spaces.
    """
    if series.dtype == 'object':
        series = series.astype(str)
        series = series.str.lower()
        series = series.str.strip()
        series = series.apply(lambda x: re.sub(r'\s+', ' ', x))
    return series

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def clean_dataframe(df, text_columns=None, deduplicate_subset=None):
    """
    Apply cleaning functions to DataFrame.
    """
    df_clean = df.copy()
    
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = clean_text_column(df_clean[col])
    
    if deduplicate_subset:
        df_clean = remove_duplicates(df_clean, subset=deduplicate_subset)
    
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'name': ['  John Doe  ', 'JANE SMITH', 'John Doe', '  Alice   '],
        'email': ['john@email.com', 'jane@email.com', 'john@email.com', 'alice@email.com'],
        'notes': ['Some   text', 'OTHER text', 'some   text', 'more']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataframe(
        df, 
        text_columns=['name', 'notes'], 
        deduplicate_subset=['email']
    )
    print("\nCleaned DataFrame:")
    print(cleaned)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for column in numeric_columns:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
            df = normalize_column(df, column)
    
    df = df.dropna()
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('sample_data.csv', ['age', 'salary', 'score'])
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaning complete. Original shape: {pd.read_csv('sample_data.csv').shape}, Cleaned shape: {cleaned_data.shape}")