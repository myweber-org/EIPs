
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import re
import pandas as pd
from typing import List, Optional

def remove_duplicates(data: List[str]) -> List[str]:
    """
    Remove duplicate entries from a list while preserving order.
    """
    seen = set()
    return [item for item in data if not (item in seen or seen.add(item))]

def normalize_text(text: str, 
                   lowercase: bool = True, 
                   remove_punctuation: bool = False) -> str:
    """
    Normalize text by applying optional transformations.
    """
    if lowercase:
        text = text.lower()
    
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    return text.strip()

def clean_dataframe(df: pd.DataFrame, 
                    columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Clean a DataFrame by removing duplicate rows and normalizing specified columns.
    """
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    if columns:
        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).apply(
                    lambda x: normalize_text(x, lowercase=True, remove_punctuation=True)
                )
    
    return df_clean

def validate_email(email: str) -> bool:
    """
    Validate email format using regex pattern.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def filter_valid_emails(emails: List[str]) -> List[str]:
    """
    Filter a list of emails, returning only valid ones.
    """
    return [email for email in emails if validate_email(email)]