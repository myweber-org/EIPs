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