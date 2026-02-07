import re
import pandas as pd

def normalize_string(text):
    if not isinstance(text, str):
        return text
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_duplicates(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    df_clean = df.drop_duplicates(subset=[column], keep='first')
    return df_clean

def fill_missing_values(df, column, fill_value):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    df_filled = df.copy()
    df_filled[column] = df_filled[column].fillna(fill_value)
    return df_filled

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email)) if isinstance(email, str) else False

def clean_dataframe(df, string_columns=None):
    df_clean = df.copy()
    if string_columns is None:
        string_columns = df_clean.select_dtypes(include=['object']).columns
    for col in string_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(normalize_string)
    return df_clean