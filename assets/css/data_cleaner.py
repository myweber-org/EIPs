import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_stats = {}
        
    def remove_outliers_iqr(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        self.original_stats[column] = {
            'min': self.df[column].min(),
            'max': self.df[column].max(),
            'rows_removed': len(self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)])
        }
        
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
        
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            self.df[column] = (self.df[column] - mean_val) / std_val
        return self
        
    def fill_missing(self, column, strategy='mean'):
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = strategy
            
        self.df[column].fillna(fill_value, inplace=True)
        return self
        
    def get_cleaned_data(self):
        return self.df.copy()
        
    def get_statistics(self):
        return self.original_stats

def clean_dataset(df, config):
    cleaner = DataCleaner(df)
    
    for column, operations in config.items():
        if 'remove_outliers' in operations:
            cleaner.remove_outliers_iqr(column)
        if 'normalize' in operations:
            cleaner.normalize_column(column, operations['normalize'])
        if 'fill_missing' in operations:
            cleaner.fill_missing(column, operations['fill_missing'])
    
    return cleaner.get_cleaned_data(), cleaner.get_statistics()