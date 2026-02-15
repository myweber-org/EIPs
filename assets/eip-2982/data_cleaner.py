
import pandas as pd
import numpy as np

def clean_column_names(df):
    """
    Standardize column names: lowercase, replace spaces with underscores.
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def fill_missing_with_median(df, column):
    """
    Fill missing values in a specified column with its median.
    """
    if column in df.columns:
        median_val = df[column].median()
        df[column].fillna(median_val, inplace=True)
    return df

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a column using the IQR method.
    """
    if column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    """
    if column in df.columns:
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
        else:
            df[column] = 0
    return df

def clean_dataset(df, config):
    """
    Apply a series of cleaning operations based on a configuration dictionary.
    """
    df = clean_column_names(df)

    for col in config.get('fill_median', []):
        df = fill_missing_with_median(df, col)

    for col in config.get('remove_outliers', []):
        df = remove_outliers_iqr(df, col)

    for col in config.get('normalize', []):
        df = normalize_column(df, col)

    return df
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result