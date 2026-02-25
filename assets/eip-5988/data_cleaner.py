import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean' and self.numeric_columns:
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        elif strategy == 'median' and self.numeric_columns:
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in self.df.columns:
                self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else fill_value, inplace=True)
        elif fill_value is not None:
            self.df.fillna(fill_value, inplace=True)
        return self
    
    def remove_outliers(self, method='zscore', threshold=3):
        if method == 'zscore' and self.numeric_columns:
            z_scores = np.abs(stats.zscore(self.df[self.numeric_columns]))
            self.df = self.df[(z_scores < threshold).all(axis=1)]
        elif method == 'iqr' and self.numeric_columns:
            for col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def summary(self):
        print(f"Original shape: {self.df.shape}")
        print(f"Numeric columns: {self.numeric_columns}")
        print(f"Categorical columns: {self.categorical_columns}")
        print(f"Missing values after cleaning:")
        print(self.df.isnull().sum())
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result