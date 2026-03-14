
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.numeric_columns
            
        clean_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)
                clean_df = clean_df[mask]
                
        return clean_df.reset_index(drop=True)
    
    def zscore_normalize(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        normalized_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val > 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
                    
        return normalized_df
    
    def minmax_normalize(self, columns=None, feature_range=(0, 1)):
        if columns is None:
            columns = self.numeric_columns
            
        normalized_df = self.df.copy()
        min_val, max_val = feature_range
        
        for col in columns:
            if col in self.numeric_columns:
                col_min = normalized_df[col].min()
                col_max = normalized_df[col].max()
                col_range = col_max - col_min
                
                if col_range > 0:
                    normalized_df[col] = (normalized_df[col] - col_min) / col_range
                    normalized_df[col] = normalized_df[col] * (max_val - min_val) + min_val
                    
        return normalized_df
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        filled_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns and filled_df[col].isnull().any():
                median_val = filled_df[col].median()
                filled_df[col] = filled_df[col].fillna(median_val)
                
        return filled_df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': list(self.numeric_columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary