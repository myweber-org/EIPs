
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
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                    
        return self
    
    def detect_outliers_zscore(self, threshold=3, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        outliers = {}
        for col in columns:
            if col in self.df.columns:
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outlier_indices = np.where(z_scores > threshold)[0]
                outliers[col] = outlier_indices.tolist()
                
        return outliers
    
    def remove_outliers_iqr(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                
        return self
    
    def normalize_data(self, method='minmax', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        normalized_df = self.df.copy()
        
        for col in columns:
            if col in normalized_df.columns:
                if method == 'minmax':
                    min_val = normalized_df[col].min()
                    max_val = normalized_df[col].max()
                    if max_val != min_val:
                        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                        
                elif method == 'zscore':
                    mean_val = normalized_df[col].mean()
                    std_val = normalized_df[col].std()
                    if std_val != 0:
                        normalized_df[col] = (normalized_df[col] - mean_val) / std_val
                        
        self.df = normalized_df
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_cleaning_report(self):
        final_shape = self.df.shape
        report = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': final_shape[0],
            'cleaned_columns': final_shape[1],
            'rows_removed': self.original_shape[0] - final_shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }
        return report

def example_usage():
    data = {
        'A': [1, 2, 2, 4, 5, 6, 100],
        'B': [10, 20, 20, 40, 50, 60, 70],
        'C': [100, 200, 200, 400, 500, 600, 700]
    }
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    cleaned_df = (cleaner
                 .remove_duplicates()
                 .handle_missing_values(strategy='mean')
                 .remove_outliers_iqr()
                 .normalize_data(method='minmax')
                 .get_cleaned_data())
    
    report = cleaner.get_cleaning_report()
    return cleaned_df, report

if __name__ == "__main__":
    result_df, summary = example_usage()
    print("Cleaned Data:")
    print(result_df)
    print("\nCleaning Report:")
    for key, value in summary.items():
        print(f"{key}: {value}")