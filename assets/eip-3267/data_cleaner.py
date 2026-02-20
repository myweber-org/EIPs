
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
        self.df = self.df[mask]
        removed_count = self.original_shape[0] - self.df.shape[0]
        return removed_count
    
    def zscore_normalize(self, column):
        mean = self.df[column].mean()
        std = self.df[column].std()
        self.df[f'{column}_normalized'] = (self.df[column] - mean) / std
        return self.df
    
    def minmax_normalize(self, column, feature_range=(0, 1)):
        min_val = self.df[column].min()
        max_val = self.df[column].max()
        
        if max_val == min_val:
            self.df[f'{column}_scaled'] = 0
        else:
            self.df[f'{column}_scaled'] = (self.df[column] - min_val) / (max_val - min_val)
            
            if feature_range != (0, 1):
                new_min, new_max = feature_range
                self.df[f'{column}_scaled'] = self.df[f'{column}_scaled'] * (new_max - new_min) + new_min
        
        return self.df
    
    def handle_missing_mean(self, column):
        mean_value = self.df[column].mean()
        self.df[column].fillna(mean_value, inplace=True)
        return self.df
    
    def handle_missing_median(self, column):
        median_value = self.df[column].median()
        self.df[column].fillna(median_value, inplace=True)
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary
    
    def get_cleaned_data(self):
        return self.df.copy()