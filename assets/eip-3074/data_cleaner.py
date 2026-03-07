
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.original_shape = data.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        clean_data = self.data.copy()
        for col in columns:
            Q1 = np.percentile(clean_data[col], 25)
            Q3 = np.percentile(clean_data[col], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)
            clean_data = clean_data[mask]
        
        removed_count = len(self.data) - len(clean_data)
        print(f"Removed {removed_count} outliers using IQR method")
        return clean_data
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        clean_data = self.data.copy()
        for col in columns:
            z_scores = np.abs(stats.zscore(clean_data[col]))
            mask = z_scores < threshold
            clean_data = clean_data[mask]
        
        removed_count = len(self.data) - len(clean_data)
        print(f"Removed {removed_count} outliers using Z-score method")
        return clean_data
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        normalized_data = self.data.copy()
        for col in columns:
            min_val = normalized_data[col].min()
            max_val = normalized_data[col].max()
            if max_val != min_val:
                normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        
        return normalized_data
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        normalized_data = self.data.copy()
        for col in columns:
            mean_val = normalized_data[col].mean()
            std_val = normalized_data[col].std()
            if std_val > 0:
                normalized_data[col] = (normalized_data[col] - mean_val) / std_val
        
        return normalized_data
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        cleaned_data = self.data.copy()
        for col in columns:
            if cleaned_data[col].isnull().any():
                if strategy == 'mean':
                    fill_value = cleaned_data[col].mean()
                elif strategy == 'median':
                    fill_value = cleaned_data[col].median()
                elif strategy == 'mode':
                    fill_value = cleaned_data[col].mode()[0]
                elif strategy == 'drop':
                    cleaned_data = cleaned_data.dropna(subset=[col])
                    continue
                else:
                    fill_value = 0
                
                cleaned_data[col] = cleaned_data[col].fillna(fill_value)
                print(f"Filled missing values in column {col} using {strategy} strategy")
        
        return cleaned_data
    
    def get_summary(self):
        summary = {
            'original_samples': self.original_shape[0],
            'original_features': self.original_shape[1],
            'current_samples': self.data.shape[0],
            'current_features': self.data.shape[1],
            'missing_values': self.data.isnull().sum().sum() if hasattr(self.data, 'isnull') else 0
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000)
    })
    
    data.iloc[10:15, 0] = np.nan
    data.iloc[20:25, 1] = np.nan
    data.iloc[5, 0] = 500
    data.iloc[8, 1] = 1000
    
    return data

if __name__ == "__main__":
    sample_data = create_sample_data()
    cleaner = DataCleaner(sample_data)
    
    print("Data Summary:")
    print(cleaner.get_summary())
    
    cleaned_data = cleaner.handle_missing_values(strategy='mean')
    cleaned_data = cleaner.remove_outliers_zscore(threshold=3)
    normalized_data = cleaner.normalize_minmax()
    
    print("\nCleaned Data Shape:", cleaned_data.shape)
    print("Normalized Data Statistics:")
    print(normalized_data.describe())