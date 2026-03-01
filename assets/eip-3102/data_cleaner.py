
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        print(f"Removed {removed_count} outliers using IQR method")
        self.df = df_clean
        return self
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns:
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                mask = z_scores < threshold
                df_clean = df_clean[mask]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        print(f"Removed {removed_count} outliers using Z-score method")
        self.df = df_clean
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        print(f"Applied Min-Max normalization to {len(columns)} columns")
        self.df = df_normalized
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    df_normalized[col] = (self.df[col] - mean_val) / std_val
        
        print(f"Applied Z-score normalization to {len(columns)} columns")
        self.df = df_normalized
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                mean_val = self.df[col].mean()
                df_filled[col] = self.df[col].fillna(mean_val)
                print(f"Filled missing values in '{col}' with mean: {mean_val:.4f}")
        
        self.df = df_filled
        return self
    
    def get_cleaned_data(self):
        return self.df
    
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
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    indices = np.random.choice(df.index, size=50, replace=False)
    df.loc[indices, 'feature_a'] = np.random.uniform(200, 300, 50)
    
    missing_indices = np.random.choice(df.index, size=30, replace=False)
    df.loc[missing_indices, 'feature_b'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    
    cleaner = DataCleaner(sample_df)
    
    cleaned_data = (cleaner
                   .remove_outliers_iqr(['feature_a', 'feature_b'])
                   .fill_missing_mean()
                   .normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
                   .get_cleaned_data())
    
    summary = cleaner.get_summary()
    print("\nCleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_data.head())
import pandas as pd
import numpy as np
from typing import Optional, List

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = 'mean',
    columns_to_drop: Optional[List[str]] = None,
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and dropping specified columns.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path where cleaned CSV will be saved
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'constant')
        columns_to_drop: List of column names to drop
        fill_value: Value to use when missing_strategy is 'constant'
    
    Returns:
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(input_path)
        
        original_shape = df.shape
        print(f"Original data shape: {original_shape}")
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors='ignore')
            print(f"Dropped columns: {columns_to_drop}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                if missing_strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif missing_strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif missing_strategy == 'mode':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif missing_strategy == 'constant' and fill_value is not None:
                    df[col].fillna(fill_value, inplace=True)
                else:
                    df[col].fillna(0, inplace=True)
        
        df.to_csv(output_path, index=False)
        
        final_shape = df.shape
        print(f"Cleaned data shape: {final_shape}")
        print(f"Rows removed: {original_shape[0] - final_shape[0]}")
        print(f"Columns removed: {original_shape[1] - final_shape[1]}")
        print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        raise
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        raise

def validate_dataframe(df: pd.DataFrame, min_rows: int = 1) -> bool:
    """
    Validate DataFrame meets minimum requirements.
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required
    
    Returns:
        Boolean indicating if DataFrame is valid
    """
    if df.empty:
        print("DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"DataFrame has fewer than {min_rows} rows")
        return False
    
    if df.isnull().sum().sum() > 0:
        print("Warning: DataFrame contains missing values")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, np.nan, 7, 8, 9],
        'C': ['x', 'y', 'z', 'x', 'y'],
        'D': [10, 11, 12, np.nan, 14]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_input.csv', index=False)
    
    cleaned_df = clean_csv_data(
        input_path='test_input.csv',
        output_path='test_output.csv',
        missing_strategy='mean',
        columns_to_drop=['C']
    )
    
    print("Data cleaning completed successfully")