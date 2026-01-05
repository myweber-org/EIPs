
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.cleaned_data = None
        
    def remove_outliers_iqr(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]
    
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.data[column]))
        return self.data[z_scores < threshold]
    
    def normalize_minmax(self, column):
        min_val = self.data[column].min()
        max_val = self.data[column].max()
        self.data[column + '_normalized'] = (self.data[column] - min_val) / (max_val - min_val)
        return self.data
    
    def standardize_zscore(self, column):
        mean_val = self.data[column].mean()
        std_val = self.data[column].std()
        self.data[column + '_standardized'] = (self.data[column] - mean_val) / std_val
        return self.data
    
    def handle_missing_mean(self, column):
        self.data[column].fillna(self.data[column].mean(), inplace=True)
        return self.data
    
    def handle_missing_median(self, column):
        self.data[column].fillna(self.data[column].median(), inplace=True)
        return self.data
    
    def get_summary(self):
        summary = {
            'original_shape': self.data.shape,
            'cleaned_shape': self.cleaned_data.shape if self.cleaned_data is not None else None,
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict()
        }
        return summary

def clean_dataset(df, outlier_method='iqr', normalize=True):
    cleaner = DataCleaner(df)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            df = cleaner.remove_outliers_iqr(col)
        elif outlier_method == 'zscore':
            df = cleaner.remove_outliers_zscore(col)
        
        if normalize:
            df = cleaner.normalize_minmax(col)
    
    cleaner.cleaned_data = df
    return cleaner