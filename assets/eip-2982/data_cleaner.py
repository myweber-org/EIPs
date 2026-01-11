
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

def clean_dataset(dataframe, numeric_columns):
    cleaned_df = dataframe.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col + '_normalized'] = normalize_minmax(cleaned_df, col)
            cleaned_df[col + '_standardized'] = standardize_zscore(cleaned_df, col)
    return cleaned_df

def generate_summary(cleaned_dataframe, original_dataframe, numeric_columns):
    summary = {}
    for col in numeric_columns:
        if col in original_dataframe.columns and col in cleaned_dataframe.columns:
            summary[col] = {
                'original_count': len(original_dataframe),
                'cleaned_count': len(cleaned_dataframe),
                'removed_outliers': len(original_dataframe) - len(cleaned_dataframe),
                'original_mean': original_dataframe[col].mean(),
                'cleaned_mean': cleaned_dataframe[col].mean(),
                'original_std': original_dataframe[col].std(),
                'cleaned_std': cleaned_dataframe[col].std()
            }
    return pd.DataFrame.from_dict(summary, orient='index')