
import re
import pandas as pd
from typing import List, Optional

def remove_special_chars(text: str) -> str:
    """Remove non-alphanumeric characters from a string."""
    if not isinstance(text, str):
        return text
    return re.sub(r'[^A-Za-z0-9\s]+', '', text)

def normalize_whitespace(text: str) -> str:
    """Replace multiple whitespace characters with a single space."""
    if not isinstance(text, str):
        return text
    return re.sub(r'\s+', ' ', text).strip()

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names by removing special characters and normalizing."""
    df.columns = [remove_special_chars(str(col)) for col in df.columns]
    df.columns = [normalize_whitespace(col) for col in df.columns]
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df

def drop_missing_rows(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Drop rows with missing values in specified columns or all columns."""
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()

def convert_to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert specified columns to numeric, coercing errors to NaN."""
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    return df_copy

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a series of cleaning functions to a DataFrame."""
    df = clean_column_names(df)
    df = drop_missing_rows(df)
    return df