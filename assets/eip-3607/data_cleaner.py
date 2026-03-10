
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: Dictionary to rename columns (optional)
        drop_duplicates: Boolean to remove duplicate rows
        normalize_text: Boolean to normalize text columns
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
    
    return cleaned_df

def _normalize_string(text):
    """Normalize string by converting to lowercase and removing extra whitespace."""
    if pd.isna(text):
        return text
    
    normalized = str(text).strip().lower()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    
    Args:
        df: pandas DataFrame
        email_column: Name of the column containing email addresses
    
    Returns:
        DataFrame with validation results
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validation_results = df.copy()
    validation_results['is_valid_email'] = validation_results[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
    )
    
    valid_count = validation_results['is_valid_email'].sum()
    total_count = len(validation_results)
    
    print(f"Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    
    return validation_results
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict, optional): Dictionary mapping old column names to new ones
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns (strip, lower case)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Rename columns if mapping provided
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Remove duplicate rows
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    # Normalize text columns
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    return cleaned_df

def remove_special_characters(text, keep_pattern=r'[a-zA-Z0-9\s]'):
    """
    Remove special characters from text, keeping only specified pattern.
    
    Args:
        text (str): Input text
        keep_pattern (str): Regex pattern of characters to keep
    
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return text
    return re.sub(f'[^{keep_pattern}]', '', str(text))

def validate_email(email):
    """
    Validate email format using regex pattern.
    
    Args:
        email (str): Email address to validate
    
    Returns:
        bool: True if email format is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email))) if pd.notna(email) else False

def fill_missing_with_mode(df, columns=None):
    """
    Fill missing values in specified columns with mode (most frequent value).
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of columns to fill. If None, fills all columns.
    
    Returns:
        pd.DataFrame: DataFrame with filled missing values
    """
    filled_df = df.copy()
    
    if columns is None:
        columns = filled_df.columns
    
    for col in columns:
        if filled_df[col].dtype == 'object':
            mode_value = filled_df[col].mode()
            if not mode_value.empty:
                filled_df[col] = filled_df[col].fillna(mode_value.iloc[0])
    
    return filled_df
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to remove duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. 
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
    
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
            for column in cleaned_df.columns:
                if cleaned_df[column].dtype == 'object':
                    mode_value = cleaned_df[column].mode()
                    if not mode_value.empty:
                        cleaned_df[column] = cleaned_df[column].fillna(mode_value[0])
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame structure and content.
    
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
    
    return True, "Data validation passed"import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing data in a DataFrame using specified strategy.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for handling missing values. 
                       Options: 'mean', 'median', 'mode', 'drop', 'fill'
        columns (list): List of columns to apply cleaning. If None, applies to all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=columns)
    elif strategy == 'mean':
        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    elif strategy == 'median':
        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif strategy == 'mode':
        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    elif strategy == 'fill':
        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)
    
    return df_clean

def detect_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to check for outliers
        threshold (float): IQR multiplier threshold
    
    Returns:
        dict: Dictionary with outlier counts per column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers[col] = outlier_mask.sum()
    
    return outliers

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize data in specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to normalize
        method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
        pd.DataFrame: Normalized DataFrame
    """
    df_norm = df.copy()
    
    if columns is None:
        columns = df_norm.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df_norm.columns:
            if method == 'minmax':
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val != min_val:
                    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df_norm[col].mean()
                std_val = df_norm[col].std()
                if std_val != 0:
                    df_norm[col] = (df_norm[col] - mean_val) / std_val
    
    return df_norm

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, 20, 30, np.nan, 50],
        'C': [100, 200, 300, 400, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_missing_data(df, strategy='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    outliers = detect_outliers_iqr(cleaned_df)
    print("\nOutlier counts:")
    print(outliers)
    
    normalized_df = normalize_data(cleaned_df, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized_df)
    
    is_valid, message = validate_dataframe(normalized_df, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {is_valid} - {message}")