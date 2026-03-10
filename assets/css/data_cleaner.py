
import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def remove_outliers(data, column, threshold=1.5):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, column, strategy='mean'):
    if strategy == 'mean':
        fill_value = data[column].mean()
    elif strategy == 'median':
        fill_value = data[column].median()
    elif strategy == 'mode':
        fill_value = data[column].mode()[0]
    else:
        fill_value = 0
    filled_data = data[column].fillna(fill_value)
    return filled_data

def clean_dataset(data, numeric_columns, outlier_threshold=1.5, normalize=True, standardize=False, missing_strategy='mean'):
    cleaned_data = data.copy()
    for col in numeric_columns:
        cleaned_data = remove_outliers(cleaned_data, col, outlier_threshold)
        cleaned_data[col] = handle_missing_values(cleaned_data, col, missing_strategy)
        if normalize:
            cleaned_data[f'{col}_normalized'] = normalize_minmax(cleaned_data, col)
        if standardize:
            cleaned_data[f'{col}_standardized'] = standardize_zscore(cleaned_data, col)
    return cleaned_data