import pandas as pd

def clean_dataset(df):
    """
    Cleans a pandas DataFrame by removing duplicate rows and
    filling missing numeric values with the column median.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()

    # Fill missing numeric values with column median
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())

    return df_cleaned

def get_summary_statistics(df):
    """
    Returns basic summary statistics for numeric columns.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    return df[numeric_cols].describe()