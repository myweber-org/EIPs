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