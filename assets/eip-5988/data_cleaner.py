
import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data_series, threshold=1.5):
    q1 = data_series.quantile(0.25)
    q3 = data_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return data_series[(data_series < lower_bound) | (data_series > upper_bound)]

def remove_outliers(df, column, method='iqr', threshold=1.5):
    if method == 'iqr':
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(df[column]))
        filtered_df = df[z_scores < threshold]
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    return filtered_df

def normalize_column(df, column, method='minmax'):
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    elif method == 'standard':
        mean_val = df[column].mean()
        std_val = df[column].std()
        df[column + '_normalized'] = (df[column] - mean_val) / std_val
    else:
        raise ValueError("Method must be 'minmax' or 'standard'")
    return df

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='standard'):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if outlier_method:
            cleaned_df = remove_outliers(cleaned_df, col, method=outlier_method)
        if normalize_method:
            cleaned_df = normalize_column(cleaned_df, col, method=normalize_method)
    return cleaned_df