
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def detect_outliers_iqr(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers
    
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        self.df = self.df[(z_scores < threshold) | (self.df[column].isna())]
        return self
    
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'standard':
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
        print(f"Original shape: {self.original_shape}")
        print(f"Cleaned shape: {self.df.shape}")
        print(f"Rows removed: {self.original_shape[0] - self.df.shape[0]}")
        return self.df
    
    def summary(self):
        summary_stats = self.df.describe(include='all')
        missing_values = self.df.isnull().sum()
        print("Summary Statistics:")
        print(summary_stats)
        print("\nMissing Values:")
        print(missing_values)
        return summary_stats, missing_values

def example_usage():
    data = {
        'age': [25, 30, 35, 200, 28, 32, 150, 29, np.nan, 31],
        'income': [50000, 60000, 55000, 1000000, 52000, 58000, 900000, 54000, 56000, 57000],
        'score': [85, 90, 88, 30, 86, 92, 25, 87, 89, 91]
    }
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    print("Detecting outliers in 'income':")
    outliers = cleaner.detect_outliers_iqr('income')
    print(outliers)
    
    cleaner.remove_outliers_zscore('income')
    cleaner.normalize_column('score', method='minmax')
    cleaner.fill_missing('age', strategy='mean')
    
    cleaned_df = cleaner.get_cleaned_data()
    print("\nCleaned Data:")
    print(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    example_usage()