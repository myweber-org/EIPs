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
        return self.df
    
    def summary(self):
        print(f"Original shape: {len(self.df)} rows, {len(self.df.columns)} columns")
        print(f"Missing values:")
        print(self.df.isnull().sum())
        print(f"Data types:")
        print(self.df.dtypes)

def clean_dataset(df, outlier_method='iqr', normalize_method='minmax'):
    cleaner = DataCleaner(df)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaner.remove_outliers_iqr(col)
        elif outlier_method == 'zscore':
            cleaner.remove_outliers_zscore(col)
        
        if normalize_method == 'minmax':
            cleaner.normalize_minmax(col)
        elif normalize_method == 'zscore':
            cleaner.normalize_zscore(col)
    
    cleaner.drop_duplicates()
    
    for col in df.columns:
        if df[col].isnull().any():
            cleaner.fill_missing_mean(col)
    
    return cleaner.get_cleaned_data()

if __name__ == "__main__":
    sample_data = {
        'feature1': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, outlier_method='iqr', normalize_method='minmax')
    print("\nCleaned DataFrame:")
    print(cleaned_df)import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers using IQR method
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_standard(data, column):
    """
    Normalize data using Standardization (Z-score normalization)
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, method='iqr', normalize=True):
    """
    Clean dataset by removing outliers and optionally normalizing
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize:
            cleaned_df = normalize_standard(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that required columns exist and have no null values
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found in columns: {null_counts[null_counts > 0].to_dict()}")
    
    return True

def get_data_summary(df):
    """
    Generate summary statistics for the dataset
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    return summaryimport pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.columns
            
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean' and self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median' and self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    self.df.dropna(subset=[col], inplace=True)
                else:
                    self.df[col].fillna(0, inplace=True)
        return self
        
    def convert_types(self, type_map: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_map.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col])
                    elif dtype == 'category':
                        self.df[col] = self.df[col].astype('category')
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert column {col} to {dtype}: {e}")
        return self
        
    def remove_outliers(self, column: str, method: str = 'iqr', threshold: float = 1.5) -> 'DataCleaner':
        if column not in self.df.columns:
            return self
            
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
        
    def normalize_column(self, column: str, method: str = 'minmax') -> 'DataCleaner':
        if column not in self.df.columns:
            return self
            
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
                
        elif method == 'standard':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val != 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
                
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_cleaning_report(self) -> Dict:
        return {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }

def clean_csv_file(input_path: str, output_path: str, cleaning_steps: Dict) -> Dict:
    try:
        df = pd.read_csv(input_path)
        cleaner = DataCleaner(df)
        
        if 'missing_values' in cleaning_steps:
            cleaner.handle_missing_values(**cleaning_steps['missing_values'])
            
        if 'convert_types' in cleaning_steps:
            cleaner.convert_types(cleaning_steps['convert_types'])
            
        if 'remove_outliers' in cleaning_steps:
            for outlier_config in cleaning_steps['remove_outliers']:
                cleaner.remove_outliers(**outlier_config)
                
        if 'normalize' in cleaning_steps:
            for normalize_config in cleaning_steps['normalize']:
                cleaner.normalize_column(**normalize_config)
        
        cleaned_df = cleaner.get_cleaned_data()
        cleaned_df.to_csv(output_path, index=False)
        
        report = cleaner.get_cleaning_report()
        report['output_file'] = output_path
        
        return report
        
    except Exception as e:
        return {'error': str(e), 'success': False}