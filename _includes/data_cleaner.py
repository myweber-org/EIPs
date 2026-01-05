
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