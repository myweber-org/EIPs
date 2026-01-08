import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from a DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in specified columns using a given strategy.
    """
    df_filled = df.copy()
    if columns is None:
        columns = df.columns

    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = 0
            df_filled[col].fillna(fill_value, inplace=True)
        else:
            df_filled[col].fillna('Unknown', inplace=True)
    return df_filled

def validate_data_types(df, schema):
    """
    Validate that DataFrame columns match expected data types.
    """
    errors = []
    for column, expected_type in schema.items():
        if column not in df.columns:
            errors.append(f"Column '{column}' not found in DataFrame.")
        elif not np.issubdtype(df[column].dtype, expected_type):
            errors.append(f"Column '{column}' has incorrect type. Expected {expected_type}, got {df[column].dtype}.")
    return errors

def normalize_column(df, column):
    """
    Normalize a numeric column to range [0, 1].
    """
    if column in df.columns and np.issubdtype(df[column].dtype, np.number):
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
        else:
            df[column] = 0
    return df

def filter_outliers_iqr(df, column, multiplier=1.5):
    """
    Filter outliers from a column using the Interquartile Range (IQR) method.
    """
    if column in df.columns and np.issubdtype(df[column].dtype, np.number):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return filtered_df
    return df