import re
import pandas as pd

def normalize_string(text):
    if not isinstance(text, str):
        return text
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_duplicates(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    df_clean = df.drop_duplicates(subset=[column], keep='first')
    return df_clean

def fill_missing_values(df, column, fill_value):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    df_filled = df.copy()
    df_filled[column] = df_filled[column].fillna(fill_value)
    return df_filled

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email)) if isinstance(email, str) else False

def clean_dataframe(df, string_columns=None):
    df_clean = df.copy()
    if string_columns is None:
        string_columns = df_clean.select_dtypes(include=['object']).columns
    for col in string_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(normalize_string)
    return df_clean
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
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
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

def save_cleaned_data(df, input_path, suffix="_cleaned"):
    """
    Save cleaned DataFrame to a new CSV file.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame
    input_path (str): Original file path
    suffix (str): Suffix to add to filename
    
    Returns:
    str: Path to saved file
    """
    if input_path.endswith('.csv'):
        output_path = input_path.replace('.csv', f'{suffix}.csv')
    else:
        output_path = f"{input_path}{suffix}.csv"
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 1, 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    cleaned_data = clean_numeric_data(sample_data)
    print("Cleaned data shape:", cleaned_data.shape)