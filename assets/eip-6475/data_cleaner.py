
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col not in self.numeric_columns:
                continue
                
            Q1 = clean_df[col].quantile(0.25)
            Q3 = clean_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)
            clean_df = clean_df[mask]
        
        return clean_df
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col not in self.numeric_columns:
                continue
                
            z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
            mask = z_scores < threshold
            clean_df = clean_df[mask]
        
        return clean_df
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col not in self.numeric_columns:
                continue
                
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            
            if max_val != min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        
        return normalized_df
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col not in self.numeric_columns:
                continue
                
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            
            if std_val > 0:
                normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        
        return normalized_df
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col not in self.numeric_columns:
                continue
                
            mean_val = filled_df[col].mean()
            filled_df[col] = filled_df[col].fillna(mean_val)
        
        return filled_df
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col not in self.numeric_columns:
                continue
                
            median_val = filled_df[col].median()
            filled_df[col] = filled_df[col].fillna(median_val)
        
        return filled_df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': self.numeric_columns,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[10:15, 'feature_a'] = np.nan
    df.loc[5, 'feature_b'] = 1000
    df.loc[95, 'feature_b'] = -500
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original data shape:", cleaner.df.shape)
    print("\nMissing values:")
    print(cleaner.df.isnull().sum())
    
    cleaned_df = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print("\nAfter IQR outlier removal:", cleaned_df.shape)
    
    normalized_df = cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    print("\nAfter min-max normalization:")
    print(normalized_df[['feature_a', 'feature_b', 'feature_c']].describe())
    
    filled_df = cleaner.fill_missing_mean(['feature_a'])
    print("\nAfter filling missing values:", filled_df.isnull().sum()['feature_a'])