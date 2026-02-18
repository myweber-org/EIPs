import re
import pandas as pd
from typing import List, Optional, Union

def clean_string(text: str, remove_digits: bool = False) -> str:
    """Clean a string by removing extra whitespace and optionally digits."""
    if not isinstance(text, str):
        return ''
    cleaned = re.sub(r'\s+', ' ', text.strip())
    if remove_digits:
        cleaned = re.sub(r'\d+', '', cleaned)
    return cleaned

def validate_email(email: str) -> bool:
    """Validate an email address format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email)) if isinstance(email, str) else False

def normalize_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Normalize a DataFrame column to lowercase and strip whitespace."""
    if column_name in df.columns:
        df[column_name] = df[column_name].astype(str).str.lower().str.strip()
    return df

def remove_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Remove outliers from a pandas Series using the IQR method."""
    if not pd.api.types.is_numeric_dtype(series):
        return series
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return series[(series >= lower_bound) & (series <= upper_bound)]

def fill_missing_with_median(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Fill missing values in specified columns with the column median."""
    df_filled = df.copy()
    if columns is None:
        columns = df_filled.select_dtypes(include=['number']).columns.tolist()
    for col in columns:
        if col in df_filled.columns and pd.api.types.is_numeric_dtype(df_filled[col]):
            median_val = df_filled[col].median()
            df_filled[col].fillna(median_val, inplace=True)
    return df_filled
import pandas as pd
import re

def clean_dataframe(df, column_name):
    """
    Clean a specific column in a pandas DataFrame by removing duplicates,
    stripping whitespace, and converting to lowercase.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].str.lower()
    df.drop_duplicates(subset=[column_name], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def normalize_text(text):
    """
    Normalize text by removing extra spaces and special characters.
    """
    if pd.isna(text):
        return text
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def process_file(input_path, output_path, column_to_clean='name'):
    """
    Read a CSV file, clean the specified column, and save to a new file.
    """
    df = pd.read_csv(input_path)
    df[column_to_clean] = df[column_to_clean].apply(normalize_text)
    df = clean_dataframe(df, column_to_clean)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    process_file(input_file, output_file)
import pandas as pd
import numpy as np
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

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    df = df.dropna()
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv', ['age', 'income', 'score'])
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaned. Original shape: {pd.read_csv('raw_data.csv').shape}, Cleaned shape: {cleaned_data.shape}")
def remove_duplicates_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result