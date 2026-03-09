
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def normalize_minmax(dataframe, column):
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    if max_val == min_val:
        return dataframe[column].apply(lambda x: 0.5)
    return dataframe[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def standardize_zscore(dataframe, column):
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    if std_val == 0:
        return dataframe[column].apply(lambda x: 0)
    return dataframe[column].apply(lambda x: (x - mean_val) / std_val)

def handle_missing_values(dataframe, strategy='mean'):
    df_copy = dataframe.copy()
    for column in df_copy.columns:
        if df_copy[column].isnull().any():
            if strategy == 'mean':
                fill_value = df_copy[column].mean()
            elif strategy == 'median':
                fill_value = df_copy[column].median()
            elif strategy == 'mode':
                fill_value = df_copy[column].mode()[0]
            else:
                fill_value = 0
            df_copy[column].fillna(fill_value, inplace=True)
    return df_copy

def clean_dataset(dataframe, outlier_columns=None, normalize_columns=None, standardize_columns=None, missing_strategy='mean'):
    df_clean = dataframe.copy()
    
    if outlier_columns:
        for column in outlier_columns:
            if column in df_clean.columns:
                df_clean = remove_outliers_iqr(df_clean, column)
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy)
    
    if normalize_columns:
        for column in normalize_columns:
            if column in df_clean.columns:
                df_clean[column] = normalize_minmax(df_clean, column)
    
    if standardize_columns:
        for column in standardize_columns:
            if column in df_clean.columns:
                df_clean[column] = standardize_zscore(df_clean, column)
    
    return df_clean