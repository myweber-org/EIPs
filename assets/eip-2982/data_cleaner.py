
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(exclude=[np.number]).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].mean()
            )
        elif strategy == 'median' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].median()
            )
        elif strategy == 'mode':
            for col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else fill_value)
        elif fill_value is not None:
            self.df = self.df.fillna(fill_value)
        return self

    def remove_outliers(self, method='zscore', threshold=3):
        if method == 'zscore' and self.numeric_columns.any():
            z_scores = np.abs(stats.zscore(self.df[self.numeric_columns]))
            self.df = self.df[(z_scores < threshold).all(axis=1)]
        elif method == 'iqr' and self.numeric_columns.any():
            Q1 = self.df[self.numeric_columns].quantile(0.25)
            Q3 = self.df[self.numeric_columns].quantile(0.75)
            IQR = Q3 - Q1
            self.df = self.df[~((self.df[self.numeric_columns] < (Q1 - 1.5 * IQR)) | 
                               (self.df[self.numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
        return self

    def normalize_data(self, method='minmax'):
        if method == 'minmax' and self.numeric_columns.any():
            self.df[self.numeric_columns] = (self.df[self.numeric_columns] - self.df[self.numeric_columns].min()) / \
                                           (self.df[self.numeric_columns].max() - self.df[self.numeric_columns].min())
        elif method == 'standard' and self.numeric_columns.any():
            self.df[self.numeric_columns] = (self.df[self.numeric_columns] - self.df[self.numeric_columns].mean()) / \
                                           self.df[self.numeric_columns].std()
        return self

    def get_cleaned_data(self):
        return self.df

def clean_dataset(df, missing_strategy='mean', outlier_method=None, normalize=False):
    cleaner = DataCleaner(df)
    cleaner.handle_missing_values(strategy=missing_strategy)
    
    if outlier_method:
        cleaner.remove_outliers(method=outlier_method)
    
    if normalize:
        cleaner.normalize_data()
    
    return cleaner.get_cleaned_data()