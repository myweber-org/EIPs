
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_mapping (dict): Optional dictionary to rename columns.
        drop_duplicates (bool): Whether to remove duplicate rows.
        normalize_text (bool): Whether to normalize text columns (strip, lower case).
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    
    return df_clean

def validate_email(email_series):
    """
    Validate email addresses in a pandas Series.
    
    Args:
        email_series (pd.Series): Series containing email addresses.
    
    Returns:
        pd.Series: Boolean series indicating valid emails.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return email_series.str.match(pattern, na=False)

def remove_special_characters(text_series, keep_chars='a-zA-Z0-9\s'):
    """
    Remove special characters from text data.
    
    Args:
        text_series (pd.Series): Series containing text data.
        keep_chars (str): Regex pattern of characters to keep.
    
    Returns:
        pd.Series: Cleaned text series.
    """
    pattern = f'[^{keep_chars}]'
    return text_series.str.replace(pattern, '', regex=True)import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (bool): Whether to fill missing values.
        fill_value: Value to use for filling missing data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def remove_outliers(df, column, threshold=3):
    """
    Remove outliers from a specific column using z-score method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
        threshold (float): Z-score threshold for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    from scipy import stats
    import numpy as np
    
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    mask = z_scores < threshold
    return df[mask].reset_index(drop=True)

def normalize_column(df, column):
    """
    Normalize a column to range [0, 1].
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to normalize.
    
    Returns:
        pd.Series: Normalized column values.
    """
    col_min = df[column].min()
    col_max = df[column].max()
    
    if col_max == col_min:
        return df[column]
    
    return (df[column] - col_min) / (col_max - col_min)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing summary statistics.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def main():
    # Example usage
    data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 14, 13, 12, 11]
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    print("\nSummary statistics before cleaning:")
    print(calculate_summary_statistics(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nSummary statistics after cleaning:")
    print(calculate_summary_statistics(cleaned_df, 'values'))

if __name__ == "__main__":
    main()