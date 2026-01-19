
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape

    def remove_duplicates(self, subset=None, keep='first'):
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        return self

    def handle_missing_values(self, strategy='drop', fill_value=None):
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            if fill_value is not None:
                self.df = self.df.fillna(fill_value)
            else:
                for col in self.df.columns:
                    if self.df[col].dtype in ['int64', 'float64']:
                        self.df[col] = self.df[col].fillna(self.df[col].median())
                    else:
                        self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        return self

    def remove_outliers(self, column, method='iqr', threshold=1.5):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(self.df[column]))
            self.df = self.df[z_scores < threshold]
        return self

    def standardize_column_names(self):
        self.df.columns = [col.strip().lower().replace(' ', '_') for col in self.df.columns]
        return self

    def get_cleaned_data(self):
        return self.df

    def get_cleaning_report(self):
        final_shape = self.df.shape
        rows_removed = self.original_shape[0] - final_shape[0]
        cols_removed = self.original_shape[1] - final_shape[1]
        
        report = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'final_rows': final_shape[0],
            'final_columns': final_shape[1],
            'rows_removed': rows_removed,
            'columns_removed': cols_removed,
            'duplicates_removed': self.original_shape[0] - self.df.drop_duplicates().shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }
        return report

def clean_dataset(df, remove_duplicates=True, handle_missing=True, standardize_names=True):
    cleaner = DataCleaner(df)
    
    if remove_duplicates:
        cleaner.remove_duplicates()
    
    if handle_missing:
        cleaner.handle_missing_values(strategy='fill')
    
    if standardize_names:
        cleaner.standardize_column_names()
    
    return cleaner.get_cleaned_data(), cleaner.get_cleaning_report()