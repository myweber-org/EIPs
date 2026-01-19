
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, case_sensitive=False):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    """
    df_clean = df.copy()
    
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean.drop_duplicates(inplace=True)
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows.")
    
    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_clean:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(lambda x: normalize_string(x, case_sensitive))
            print(f"Normalized column: {col}")
    
    return df_clean

def normalize_string(text, case_sensitive=False):
    """
    Normalize a string by stripping whitespace and optionally converting case.
    """
    if pd.isna(text):
        return text
    
    text = str(text).strip()
    
    if not case_sensitive:
        text = text.lower()
    
    text = re.sub(r'\s+', ' ', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].apply(lambda x: bool(re.match(pattern, str(x))) if pd.notna(x) else False)
    
    valid_count = df['email_valid'].sum()
    total_count = len(df)
    
    print(f"Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.2f}%)")
    
    return df

if __name__ == "__main__":
    sample_data = {
        'name': ['  John Doe  ', 'Jane Smith', 'John Doe', 'ALICE WONDER', '  bob  '],
        'email': ['john@example.com', 'invalid-email', 'JOHN@example.com', 'alice@test.org', 'bob@domain.co'],
        'age': [25, 30, 25, 28, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataframe(df, columns_to_clean=['name', 'email'], remove_duplicates=True, case_sensitive=False)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    validated_df = validate_email_column(cleaned_df, 'email')
    print("\nDataFrame with email validation:")
    print(validated_df[['name', 'email', 'email_valid']])import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. Can be 'mean', 'median', 
                                   'mode', or a dictionary of column:value pairs. Default is None.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
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
    
    return True, "DataFrame is valid"