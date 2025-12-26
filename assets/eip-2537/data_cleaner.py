import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        if column not in self.numeric_columns:
            raise ValueError(f"Column {column} is not numeric")
            
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
        self.df = self.df[mask]
        return self
        
    def normalize_column(self, column, method='minmax'):
        if column not in self.numeric_columns:
            raise ValueError(f"Column {column} is not numeric")
            
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val > 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        return self
        
    def fill_missing(self, column, strategy='mean'):
        if column not in self.df.columns:
            raise ValueError(f"Column {column} does not exist")
            
        if strategy == 'mean' and column in self.numeric_columns:
            fill_value = self.df[column].mean()
        elif strategy == 'median' and column in self.numeric_columns:
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = 0
            
        self.df[column] = self.df[column].fillna(fill_value)
        return self
        
    def get_cleaned_data(self):
        return self.df.copy()
        
    def summary(self):
        summary_data = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'numeric_columns': list(self.numeric_columns),
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary_data

def example_usage():
    np.random.seed(42)
    data = {
        'A': np.random.normal(100, 15, 100),
        'B': np.random.uniform(0, 1, 100),
        'C': np.random.randint(1, 100, 100)
    }
    df = pd.DataFrame(data)
    df.loc[10:15, 'A'] = np.nan
    df.loc[5, 'B'] = 999
    
    cleaner = DataCleaner(df)
    cleaner.remove_outliers_iqr('A')
    cleaner.normalize_column('B', method='minmax')
    cleaner.fill_missing('A', strategy='mean')
    
    cleaned_df = cleaner.get_cleaned_data()
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Summary: {cleaner.summary()}")
    
    return cleaned_df

if __name__ == "__main__":
    example_usage()