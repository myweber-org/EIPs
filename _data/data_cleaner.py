
import numpy as np
import pandas as pd
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
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
                
        self.df = df_clean.reset_index(drop=True)
        return self
        
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
                    
        self.df = df_normalized
        return self
        
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                median_val = self.df[col].median()
                df_filled[col] = self.df[col].fillna(median_val)
                
        self.df = df_filled
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
        
    def summary(self):
        print(f"Original rows: {self.original_shape[0]}")
        print(f"Cleaned rows: {self.df.shape[0]}")
        print(f"Rows removed: {self.get_removed_count()}")
        print(f"Columns: {self.df.shape[1]}")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("\nNumeric columns summary:")
            for col in numeric_cols[:5]:
                print(f"{col}: mean={self.df[col].mean():.3f}, std={self.df[col].std():.3f}")

def example_usage():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(1000, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(1000, 20), 'feature_b'] = 1000
    
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                 .fill_missing_median()
                 .remove_outliers_iqr(['feature_a', 'feature_b'])
                 .normalize_minmax()
                 .get_cleaned_data())
    
    cleaner.summary()
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print(f"\nCleaned data shape: {result.shape}")
    print(result.head())import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', output_filepath=None):
    """
    Load a CSV file, handle missing values, and optionally save cleaned data.
    
    Args:
        filepath (str): Path to the input CSV file.
        fill_strategy (str): Strategy for filling missing values.
                             Options: 'mean', 'median', 'zero', 'drop'.
        output_filepath (str, optional): Path to save cleaned CSV.
                                         If None, returns DataFrame.
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_filepath is None.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values.")
            
            if fill_strategy == 'mean':
                df = df.fillna(df.mean(numeric_only=True))
            elif fill_strategy == 'median':
                df = df.fillna(df.median(numeric_only=True))
            elif fill_strategy == 'zero':
                df = df.fillna(0)
            elif fill_strategy == 'drop':
                df = df.dropna()
            else:
                raise ValueError(f"Unknown fill strategy: {fill_strategy}")
            
            print(f"Missing values handled using '{fill_strategy}' strategy.")
        
        if output_filepath:
            df.to_csv(output_filepath, index=False)
            print(f"Cleaned data saved to: {output_filepath}")
            return None
        else:
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing columns {missing_cols}")
            return False
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("Warning: No numeric columns found in DataFrame.")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, np.nan, 7, 8, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned = clean_csv_data('test_data.csv', fill_strategy='mean')
    
    if cleaned is not None:
        is_valid = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
        print(f"Data validation result: {is_valid}")
        print("Cleaned data preview:")
        print(cleaned.head())
    
    import os
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')