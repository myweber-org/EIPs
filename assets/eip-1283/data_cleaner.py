
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def remove_outliers_zscore(dataframe, column, threshold=3):
    z_scores = np.abs(stats.zscore(dataframe[column]))
    return dataframe[z_scores < threshold]

def normalize_minmax(dataframe, column):
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    dataframe[column] = (dataframe[column] - min_val) / (max_val - min_val)
    return dataframe

def normalize_zscore(dataframe, column):
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    dataframe[column] = (dataframe[column] - mean_val) / std_val
    return dataframe

def clean_dataset(dataframe, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            if outlier_method == 'iqr':
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            elif outlier_method == 'zscore':
                cleaned_df = remove_outliers_zscore(cleaned_df, column)
            
            if normalize_method == 'minmax':
                cleaned_df = normalize_minmax(cleaned_df, column)
            elif normalize_method == 'zscore':
                cleaned_df = normalize_zscore(cleaned_df, column)
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(dataframe):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if dataframe.empty:
        raise ValueError("DataFrame is empty")
    
    return True

def get_data_summary(dataframe):
    summary = {
        'original_shape': dataframe.shape,
        'missing_values': dataframe.isnull().sum().to_dict(),
        'data_types': dataframe.dtypes.to_dict(),
        'numeric_columns': dataframe.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': dataframe.select_dtypes(include=['object']).columns.tolist()
    }
    return summary