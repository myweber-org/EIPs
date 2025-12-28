
import re
import pandas as pd
from typing import Optional, Union, List

def remove_duplicates(data: Union[List, pd.Series, pd.DataFrame]) -> Union[List, pd.Series, pd.DataFrame]:
    """
    Remove duplicate entries from a list, Series, or DataFrame.
    """
    if isinstance(data, list):
        return list(dict.fromkeys(data))
    elif isinstance(data, pd.Series):
        return data.drop_duplicates()
    elif isinstance(data, pd.DataFrame):
        return data.drop_duplicates()
    else:
        raise TypeError("Input must be a list, pandas Series, or pandas DataFrame")

def validate_email(email: str) -> bool:
    """
    Validate an email address using a regular expression.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def normalize_string(text: str, case: str = 'lower') -> str:
    """
    Normalize string by stripping whitespace and converting case.
    """
    text = text.strip()
    if case == 'lower':
        return text.lower()
    elif case == 'upper':
        return text.upper()
    elif case == 'title':
        return text.title()
    else:
        return text

def fill_missing_values(df: pd.DataFrame, column: str, value: Union[str, int, float]) -> pd.DataFrame:
    """
    Fill missing values in a specified column of a DataFrame.
    """
    df_copy = df.copy()
    df_copy[column] = df_copy[column].fillna(value)
    return df_copy

def convert_to_numeric(series: pd.Series, errors: str = 'coerce') -> pd.Series:
    """
    Convert a pandas Series to numeric type, handling errors as specified.
    """
    return pd.to_numeric(series, errors=errors)

def filter_by_threshold(df: pd.DataFrame, column: str, threshold: float, keep: str = 'above') -> pd.DataFrame:
    """
    Filter DataFrame rows based on a threshold value in a specified column.
    """
    if keep == 'above':
        return df[df[column] > threshold]
    elif keep == 'below':
        return df[df[column] < threshold]
    elif keep == 'equal':
        return df[df[column] == threshold]
    else:
        raise ValueError("keep parameter must be 'above', 'below', or 'equal'")

def sample_data(df: pd.DataFrame, n: int = 5, random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Return a random sample of rows from a DataFrame.
    """
    return df.sample(n=n, random_state=random_state)