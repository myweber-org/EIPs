
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].copy()
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].copy()
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values with specified strategy
    """
    if columns is None:
        columns = data.columns
    
    data_copy = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            elif strategy == 'zero':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_copy[col] = data[col].fillna(fill_value)
    
    return data_copy

def validate_dataframe(data, required_columns=None):
    """
    Validate DataFrame structure and content
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        removed_count = len(self.df) - len(df_clean)
        self.df = df_clean
        return removed_count
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                df_clean = df_clean[(z_scores < threshold) | df_clean[col].isna()]
        
        removed_count = len(self.df) - len(df_clean)
        self.df = df_clean
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                col_min = df_normalized[col].min()
                col_max = df_normalized[col].max()
                if col_max > col_min:
                    df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
        
        self.df = df_normalized
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                col_mean = df_normalized[col].mean()
                col_std = df_normalized[col].std()
                if col_std > 0:
                    df_normalized[col] = (df_normalized[col] - col_mean) / col_std
        
        self.df = df_normalized
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        original_rows, original_cols = self.original_shape
        cleaned_rows, cleaned_cols = self.df.shape
        removed_rows = original_rows - cleaned_rows
        
        summary = {
            'original_shape': self.original_shape,
            'cleaned_shape': (cleaned_rows, cleaned_cols),
            'rows_removed': removed_rows,
            'removal_percentage': (removed_rows / original_rows * 100) if original_rows > 0 else 0
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature1'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'feature2'] = 1000
    
    cleaner = DataCleaner(df)
    print(f"Original shape: {cleaner.original_shape}")
    
    outliers_removed = cleaner.remove_outliers_iqr(['feature1', 'feature2'])
    print(f"Removed {outliers_removed} outliers using IQR method")
    
    cleaner.fill_missing_mean(['feature1'])
    cleaner.normalize_minmax(['feature1', 'feature2', 'feature3'])
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print(f"Cleaned shape: {summary['cleaned_shape']}")
    print(f"Rows removed: {summary['rows_removed']}")
    print(f"Removal percentage: {summary['removal_percentage']:.2f}%")
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_data.head())