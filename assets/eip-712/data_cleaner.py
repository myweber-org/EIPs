import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.original_shape = data.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        clean_data = self.data.copy()
        for col in columns:
            Q1 = clean_data[col].quantile(0.25)
            Q3 = clean_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = (clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)
            clean_data = clean_data[mask]
        
        self.removed_count = self.original_shape[0] - clean_data.shape[0]
        return clean_data
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        normalized_data = self.data.copy()
        for col in columns:
            min_val = normalized_data[col].min()
            max_val = normalized_data[col].max()
            if max_val > min_val:
                normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        
        return normalized_data
    
    def standardize_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        standardized_data = self.data.copy()
        for col in columns:
            mean = standardized_data[col].mean()
            std = standardized_data[col].std()
            if std > 0:
                standardized_data[col] = (standardized_data[col] - mean) / std
        
        return standardized_data
    
    def handle_missing_mean(self, columns=None):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        filled_data = self.data.copy()
        for col in columns:
            if filled_data[col].isnull().any():
                mean_val = filled_data[col].mean()
                filled_data[col] = filled_data[col].fillna(mean_val)
        
        return filled_data

def create_sample_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000)
    })
    
    data.iloc[10:15, 0] = np.nan
    data.iloc[20:25, 1] = np.nan
    
    outliers = np.random.randint(0, 1000, 20)
    data.iloc[outliers, 0] = data.iloc[outliers, 0] * 5
    
    return data

if __name__ == "__main__":
    sample_data = create_sample_data()
    cleaner = DataCleaner(sample_data)
    
    print("Original data shape:", cleaner.original_shape)
    print("Missing values:", sample_data.isnull().sum().sum())
    
    cleaned = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print("After outlier removal:", cleaned.shape)
    print("Removed", cleaner.removed_count, "rows")
    
    normalized = cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    print("Normalized data range:", normalized.min().min(), "to", normalized.max().max())
    
    standardized = cleaner.standardize_zscore(['feature_a', 'feature_b'])
    print("Standardized mean ~0:", standardized['feature_a'].mean())
    print("Standardized std ~1:", standardized['feature_a'].std())
    
    filled = cleaner.handle_missing_mean()
    print("Missing values after fill:", filled.isnull().sum().sum())