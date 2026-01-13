
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, case_normalization='lower'):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    """
    df_clean = df.copy()

    if remove_duplicates:
        initial_rows = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - df_clean.shape[0]
        print(f"Removed {removed} duplicate row(s).")

    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns.tolist()

    for col in columns_to_clean:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(lambda x: normalize_string(x, case_normalization) if pd.notnull(x) else x)

    return df_clean

def normalize_string(s, normalization='lower'):
    s = str(s).strip()
    s = re.sub(r'\s+', ' ', s)

    if normalization == 'lower':
        s = s.lower()
    elif normalization == 'upper':
        s = s.upper()
    elif normalization == 'title':
        s = s.title()

    return s

def example_usage():
    data = {
        'Name': ['  alice  ', 'Bob', 'Alice', 'bob', 'CHARLIE', '  charlie  '],
        'Age': [25, 30, 25, 30, 35, 35],
        'City': ['New York', 'los angeles', 'New York', 'LOS ANGELES', 'London', 'london']
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")

    df_clean = clean_dataframe(df, columns_to_clean=['Name', 'City'], remove_duplicates=True, case_normalization='title')
    print("Cleaned DataFrame:")
    print(df_clean)

if __name__ == "__main__":
    example_usage()