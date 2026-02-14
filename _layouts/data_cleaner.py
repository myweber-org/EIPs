
import pandas as pd
import numpy as np
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
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
        
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        self.df = self.df[z_scores < threshold]
        return self
        
    def fill_missing_mean(self, column):
        self.df[column] = self.df[column].fillna(self.df[column].mean())
        return self
        
    def fill_missing_median(self, column):
        self.df[column] = self.df[column].fillna(self.df[column].median())
        return self
        
    def fill_missing_mode(self, column):
        self.df[column] = self.df[column].fillna(self.df[column].mode()[0])
        return self
        
    def drop_missing_rows(self, threshold=0.8):
        self.df = self.df.dropna(thresh=threshold * len(self.df.columns))
        return self
        
    def drop_missing_columns(self, threshold=0.8):
        self.df = self.df.dropna(axis=1, thresh=threshold * len(self.df))
        return self
        
    def standardize_column(self, column):
        mean = self.df[column].mean()
        std = self.df[column].std()
        self.df[column] = (self.df[column] - mean) / std
        return self
        
    def normalize_column(self, column):
        min_val = self.df[column].min()
        max_val = self.df[column].max()
        self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
        
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.get_removed_count(),
            'columns_removed': self.original_shape[1] - self.df.shape[1]
        }
        return summary

def example_usage():
    data = {
        'A': [1, 2, 3, 100, 5, 6, None, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    }
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    cleaned_df = (cleaner
                 .remove_outliers_iqr('A')
                 .fill_missing_mean('A')
                 .standardize_column('B')
                 .normalize_column('C')
                 .get_cleaned_data())
    
    print("Original shape:", df.shape)
    print("Cleaned shape:", cleaned_df.shape)
    print("Summary:", cleaner.get_summary())
    
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print(result.head())