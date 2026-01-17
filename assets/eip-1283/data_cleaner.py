
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
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        
    def detect_outliers_zscore(self, threshold=3):
        outliers = {}
        for col in self.numeric_columns:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outlier_indices = np.where(z_scores > threshold)[0]
            if len(outlier_indices) > 0:
                outliers[col] = {
                    'indices': outlier_indices.tolist(),
                    'count': len(outlier_indices),
                    'percentage': (len(outlier_indices) / len(self.df)) * 100
                }
        return outliers
    
    def impute_missing_median(self):
        for col in self.numeric_columns:
            if self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
        return self.df
    
    def remove_duplicates(self, subset=None, keep='first'):
        initial_count = len(self.df)
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        removed = initial_count - len(self.df)
        return self.df, removed
    
    def normalize_numeric(self, method='minmax'):
        normalized_df = self.df.copy()
        for col in self.numeric_columns:
            if method == 'minmax':
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val != 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        return normalized_df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': list(self.numeric_columns),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary