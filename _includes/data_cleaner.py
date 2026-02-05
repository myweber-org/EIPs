
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data.copy()
        self.original_shape = data.shape
        
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        clean_data = self.data.copy()
        for col in columns:
            if col in clean_data.columns:
                Q1 = clean_data[col].quantile(0.25)
                Q3 = clean_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                clean_data = clean_data[(clean_data[col] >= lower_bound) & 
                                       (clean_data[col] <= upper_bound)]
        return clean_data
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        normalized_data = self.data.copy()
        for col in columns:
            if col in normalized_data.columns:
                min_val = normalized_data[col].min()
                max_val = normalized_data[col].max()
                if max_val > min_val:
                    normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        return normalized_data
    
    def standardize_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        standardized_data = self.data.copy()
        for col in columns:
            if col in standardized_data.columns:
                mean_val = standardized_data[col].mean()
                std_val = standardized_data[col].std()
                if std_val > 0:
                    z_scores = np.abs((standardized_data[col] - mean_val) / std_val)
                    standardized_data = standardized_data[z_scores < threshold]
        return standardized_data
    
    def handle_missing(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        filled_data = self.data.copy()
        for col in columns:
            if col in filled_data.columns and filled_data[col].isnull().any():
                if strategy == 'mean':
                    fill_value = filled_data[col].mean()
                elif strategy == 'median':
                    fill_value = filled_data[col].median()
                elif strategy == 'mode':
                    fill_value = filled_data[col].mode()[0]
                else:
                    fill_value = 0
                filled_data[col].fillna(fill_value, inplace=True)
        return filled_data
    
    def get_summary(self):
        summary = {
            'original_shape': self.original_shape,
            'current_shape': self.data.shape,
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(exclude=[np.number]).columns)
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'temperature': np.random.normal(25, 5, 100),
        'humidity': np.random.uniform(30, 90, 100),
        'pressure': np.random.normal(1013, 10, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    data.loc[np.random.choice(100, 5), 'temperature'] = np.nan
    data.loc[np.random.choice(100, 3), 'pressure'] = np.nan
    return data

if __name__ == "__main__":
    sample_data = create_sample_data()
    cleaner = DataCleaner(sample_data)
    
    print("Original data shape:", cleaner.original_shape)
    print("\nMissing values:")
    print(sample_data.isnull().sum())
    
    cleaned_data = cleaner.handle_missing(strategy='mean')
    print("\nAfter handling missing values:", cleaned_data.shape)
    
    normalized_data = cleaner.normalize_minmax(['temperature', 'humidity', 'pressure'])
    print("\nAfter normalization:")
    print(normalized_data[['temperature', 'humidity', 'pressure']].describe())
    
    summary = cleaner.get_summary()
    print("\nData summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
    
    def remove_outliers_iqr(self, column, multiplier=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
    
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        self.df = self.df[z_scores < threshold]
        return self
    
    def normalize_minmax(self, column):
        min_val = self.df[column].min()
        max_val = self.df[column].max()
        self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        return self
    
    def normalize_zscore(self, column):
        mean_val = self.df[column].mean()
        std_val = self.df[column].std()
        self.df[column] = (self.df[column] - mean_val) / std_val
        return self
    
    def fill_missing_mean(self, column):
        self.df[column].fillna(self.df[column].mean(), inplace=True)
        return self
    
    def fill_missing_median(self, column):
        self.df[column].fillna(self.df[column].median(), inplace=True)
        return self
    
    def drop_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        return self
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def summary(self):
        print(f"Original shape: {self.df.shape}")
        print(f"Columns: {self.original_columns}")
        print(f"Missing values per column:")
        for col in self.df.columns:
            missing = self.df[col].isnull().sum()
            print(f"  {col}: {missing}")