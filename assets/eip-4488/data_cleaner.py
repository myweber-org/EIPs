
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
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
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
        return self.df
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_standardized = self.df.copy()
        for col in columns:
            if col in df_standardized.columns:
                mean_val = df_standardized[col].mean()
                std_val = df_standardized[col].std()
                if std_val > 0:
                    df_standardized[col] = (df_standardized[col] - mean_val) / std_val
        
        self.df = df_standardized
        return self.df
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns and df_filled[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_filled[col].mean()
                elif strategy == 'median':
                    fill_value = df_filled[col].median()
                elif strategy == 'mode':
                    fill_value = df_filled[col].mode()[0]
                else:
                    fill_value = 0
                
                df_filled[col] = df_filled[col].fillna(fill_value)
        
        self.df = df_filled
        return self.df
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    indices = np.random.choice(df.index, size=50, replace=False)
    df.loc[indices, 'feature_a'] = np.random.uniform(300, 500, 50)
    
    missing_indices = np.random.choice(df.index, size=30, replace=False)
    df.loc[missing_indices, 'feature_b'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original data shape:", cleaner.original_shape)
    print("Missing values:", sample_df.isnull().sum().sum())
    
    removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b', 'feature_c'])
    print(f"Removed {removed} outliers")
    
    cleaner.handle_missing_values(strategy='median')
    cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print("\nCleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nCleaned data statistics:")
    print(cleaned_df.describe())
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list, optional): List of numeric columns to clean.
                                         If None, uses all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column: {column}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(100),
        'value': np.random.randn(100) * 10 + 50
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[95:99, 'value'] = [200, -100, 300, 150, 250]
    
    print("Original dataset shape:", df.shape)
    print("Original statistics:")
    print(df['value'].describe())
    
    cleaned_df = clean_dataset(df, ['value'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("Cleaned statistics:")
    print(cleaned_df['value'].describe())