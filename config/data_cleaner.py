import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text by converting to lowercase, removing extra spaces,
    and stripping special characters except basic punctuation.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x))
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s.,!?-]', '', x))
    df[column_name] = df[column_name].str.strip()
    return df

def remove_duplicate_rows(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def standardize_dates(df, column_name, date_format='%Y-%m-%d'):
    """
    Attempt to standardize date column to specified format.
    """
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce').dt.strftime(date_format)
    return df

def clean_dataset(df, text_columns=None, date_columns=None, deduplicate=True):
    """
    Main cleaning function that applies multiple cleaning operations.
    """
    df_clean = df.copy()
    
    if text_columns:
        for col in text_columns:
            df_clean = clean_text_column(df_clean, col)
    
    if date_columns:
        for col in date_columns:
            df_clean = standardize_dates(df_clean, col)
    
    if deduplicate:
        df_clean = remove_duplicate_rows(df_clean)
    
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Alice Johnson  '],
        'email': ['JOHN@example.com', 'jane@test.com', 'john@example.com', 'alice@sample.org'],
        'date': ['2023-01-15', '15/02/2023', 'Jan 20, 2023', '2023.03.10']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nCleaned dataset:")
    cleaned_df = clean_dataset(df, text_columns=['name', 'email'], date_columns=['date'])
    print(cleaned_df)