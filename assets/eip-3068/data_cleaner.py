
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self
        
    def handle_missing_values(self, strategy='mean'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            for col in numeric_cols:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in numeric_cols:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in numeric_cols:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                
        return self
        
    def detect_outliers_zscore(self, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_mask = pd.Series([False] * len(self.df))
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            col_outliers = (z_scores > threshold)
            outliers_mask = outliers_mask | col_outliers.reindex(self.df.index, fill_value=False)
            
        return outliers_mask
    
    def remove_outliers(self, threshold=3):
        outliers_mask = self.detect_outliers_zscore(threshold)
        self.df = self.df[~outliers_mask]
        return self
        
    def normalize_data(self, method='minmax'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    
        elif method == 'standard':
            for col in numeric_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
                    
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
            'remaining_missing': self.df.isnull().sum().sum()
        }
        
        return report

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.randn(100),
        'feature_b': np.random.randn(100) * 2 + 5,
        'feature_c': np.random.randn(100) * 0.5 + 10
    }
    
    df = pd.DataFrame(data)
    df.iloc[5, 0] = np.nan
    df.iloc[10, 1] = np.nan
    df.iloc[15, 2] = np.nan
    
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    
    df.iloc[95, 0] = 100
    df.iloc[96, 1] = -50
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    cleaned_df = (cleaner
                 .remove_duplicates()
                 .handle_missing_values(strategy='mean')
                 .remove_outliers(threshold=3)
                 .normalize_data(method='minmax')
                 .get_cleaned_data())
    
    report = cleaner.get_cleaning_report()
    print(f"Data cleaning completed:")
    print(f"Original shape: {report['original_rows']}x{report['original_columns']}")
    print(f"Final shape: {report['final_rows']}x{report['final_columns']}")
    print(f"Rows removed: {report['rows_removed']}")
    print(f"Missing values remaining: {report['remaining_missing']}")