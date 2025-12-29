
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
    return [email for email in emails if validate_email(email)]import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (bool): Whether to fill missing values. Default is True.
    fill_value: Value to use for filling missing data. Default is 0.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10, None, 20, 30, None],
        'category': ['A', 'B', 'B', 'C', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_data(cleaned, ['id', 'value'])
    print(f"\nValidation: {is_valid} - {message}")