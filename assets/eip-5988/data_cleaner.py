
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
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                mean_val = df_normalized[col].mean()
                std_val = df_normalized[col].std()
                if std_val > 0:
                    df_normalized[col] = (df_normalized[col] - mean_val) / std_val
        
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
    
    def get_removed_percentage(self):
        original_rows = self.original_shape[0]
        current_rows = len(self.df)
        return ((original_rows - current_rows) / original_rows) * 100 if original_rows > 0 else 0

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    outliers_idx = np.random.choice(1000, 50, replace=False)
    df.loc[outliers_idx, 'feature_a'] = np.random.uniform(200, 300, 50)
    
    missing_idx = np.random.choice(1000, 100, replace=False)
    df.loc[missing_idx, 'feature_b'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print(f"Original data shape: {sample_df.shape}")
    print(f"Missing values: {sample_df.isnull().sum().sum()}")
    
    cleaner = DataCleaner(sample_df)
    
    removed_iqr = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print(f"Removed {removed_iqr} outliers using IQR method")
    
    cleaner.fill_missing_mean(['feature_b'])
    print(f"Filled missing values with mean")
    
    cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    print(f"Normalized features using min-max scaling")
    
    cleaned_df = cleaner.get_cleaned_data()
    print(f"Cleaned data shape: {cleaned_df.shape}")
    print(f"Removed percentage: {cleaner.get_removed_percentage():.2f}%")
    print(f"Cleaned data summary:\n{cleaned_df.describe()}")
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_missing(self, threshold=0.3):
        missing_percent = self.df.isnull().sum() / len(self.df)
        columns_to_drop = missing_percent[missing_percent > threshold].index
        self.df = self.df.drop(columns=columns_to_drop)
        return self
    
    def fill_numeric_missing(self, method='median'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if method == 'median':
                    fill_value = self.df[col].median()
                elif method == 'mean':
                    fill_value = self.df[col].mean()
                elif method == 'mode':
                    fill_value = self.df[col].mode()[0]
                else:
                    fill_value = 0
                
                self.df[col] = self.df[col].fillna(fill_value)
        
        return self
    
    def detect_outliers_zscore(self, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_mask = pd.Series([False] * len(self.df))
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col].fillna(self.df[col].median())))
            outlier_mask = outlier_mask | (z_scores > threshold)
        
        return outlier_mask
    
    def remove_outliers(self, threshold=3):
        outlier_mask = self.detect_outliers_zscore(threshold)
        self.df = self.df[~outlier_mask].reset_index(drop=True)
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
        rows_removed = self.original_shape[0] - len(self.df)
        cols_removed = self.original_shape[1] - self.df.shape[1]
        
        report = {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'rows_removed': rows_removed,
            'columns_removed': cols_removed,
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(exclude=[np.number]).columns)
        }
        
        return report

def clean_dataset(df, outlier_threshold=3, normalize=True):
    cleaner = DataCleaner(df)
    
    cleaner.remove_missing(threshold=0.3)
    cleaner.fill_numeric_missing(method='median')
    cleaner.remove_outliers(threshold=outlier_threshold)
    
    if normalize:
        cleaner.normalize_data(method='minmax')
    
    return cleaner.get_cleaned_data(), cleaner.get_cleaning_report()
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_method='mean'):
    """
    Clean dataset by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        columns_to_check: list of columns to check for duplicates (default: all columns)
        fill_method: method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    original_shape = df.shape
    
    # Remove duplicate rows
    if columns_to_check:
        df_cleaned = df.drop_duplicates(subset=columns_to_check)
    else:
        df_cleaned = df.drop_duplicates()
    
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    
    # Handle missing values
    if fill_method == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif fill_method in ['mean', 'median']:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_method == 'mean':
                fill_value = df_cleaned[col].mean()
            else:
                fill_value = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
    elif fill_method == 'mode':
        for col in df_cleaned.columns:
            mode_value = df_cleaned[col].mode()
            if not mode_value.empty:
                df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])
    
    missing_filled = df.isna().sum().sum() - df_cleaned.isna().sum().sum()
    
    print(f"Cleaning complete:")
    print(f"  - Removed {duplicates_removed} duplicate rows")
    print(f"  - Filled {missing_filled} missing values")
    print(f"  - Original shape: {original_shape}")
    print(f"  - Cleaned shape: {df_cleaned.shape}")
    
    return df_cleaned

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
        min_rows: minimum number of rows required
    
    Returns:
        Boolean indicating if validation passed
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Validation failed: Missing required columns: {missing_columns}")
            return False
    
    print("DataFrame validation passed")
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
        'age': [25, 30, 30, None, 35, 40],
        'score': [85.5, 92.0, 92.0, 78.5, None, 95.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, columns_to_check=['id', 'name'], fill_method='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned dataframe
    validation_passed = validate_dataframe(
        cleaned_df, 
        required_columns=['id', 'name', 'age', 'score'],
        min_rows=1
    )