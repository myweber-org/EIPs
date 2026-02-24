import re
import unicodedata

def clean_text(text, remove_digits=False, remove_punctuation=False, to_lower=True):
    """
    Clean and normalize a given text string.

    Args:
        text (str): Input text to be cleaned.
        remove_digits (bool): If True, remove all digits from the text.
        remove_punctuation (bool): If True, remove all punctuation characters.
        to_lower (bool): If True, convert text to lowercase.

    Returns:
        str: Cleaned and normalized text.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    # Normalize unicode characters (e.g., convert accented characters to their base form)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Optionally convert to lowercase
    if to_lower:
        text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Optionally remove digits
    if remove_digits:
        text = re.sub(r'\d+', '', text)

    # Optionally remove punctuation
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)

    return text

def tokenize_text(text, tokenizer=None):
    """
    Tokenize the cleaned text. If no custom tokenizer is provided, split by whitespace.

    Args:
        text (str): Cleaned text to tokenize.
        tokenizer (callable, optional): A function that takes a string and returns a list of tokens.
                                        Defaults to splitting by whitespace.

    Returns:
        list: List of tokens.
    """
    if tokenizer is None:
        tokenizer = lambda x: x.split()
    return tokenizer(text)

if __name__ == "__main__":
    # Example usage
    sample_text = "Hello, World! This is a sample text with numbers 123 and punctuation!!!"
    cleaned = clean_text(sample_text, remove_digits=True, remove_punctuation=True)
    tokens = tokenize_text(cleaned)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")
    print(f"Tokens: {tokens}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name for statistics.
    
    Returns:
    dict: Dictionary containing mean, median, std, min, and max.
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max()
    }
    return stats

if __name__ == "__main__":
    sample_data = {'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 102, 12, 14, 13, 12, 11, 10, 14, 13, 12, 11]}
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print("\nOriginal Statistics:")
    print(calculate_basic_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Statistics:")
    print(calculate_basic_stats(cleaned_df, 'values'))
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean. If None, uses all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not process column '{col}': {e}")
    
    return cleaned_df

def get_cleaning_report(original_df, cleaned_df):
    """
    Generate a report comparing original and cleaned DataFrames.
    
    Args:
        original_df (pd.DataFrame): Original DataFrame
        cleaned_df (pd.DataFrame): Cleaned DataFrame
    
    Returns:
        dict: Dictionary containing cleaning statistics
    """
    report = {
        'original_rows': len(original_df),
        'cleaned_rows': len(cleaned_df),
        'rows_removed': len(original_df) - len(cleaned_df),
        'removal_percentage': ((len(original_df) - len(cleaned_df)) / len(original_df)) * 100
    }
    
    return report

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[0, 'A'] = 1000
    
    print("Original DataFrame shape:", df.shape)
    
    cleaned_df = clean_numeric_data(df, columns=['A', 'B'])
    
    print("Cleaned DataFrame shape:", cleaned_df.shape)
    
    report = get_cleaning_report(df, cleaned_df)
    print("Cleaning Report:", report)