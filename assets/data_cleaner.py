
import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        self.df = pd.read_csv(self.file_path)
        print(f"Loaded data with shape: {self.df.shape}")
        return self
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                self.df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{col}' with {strategy}: {fill_value}")
        
        return self
    
    def remove_duplicates(self, subset=None):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        initial_count = len(self.df)
        self.df.drop_duplicates(subset=subset, inplace=True, keep='first')
        removed = initial_count - len(self.df)
        
        if removed > 0:
            print(f"Removed {removed} duplicate rows")
        
        return self
    
    def normalize_numeric(self, columns=None):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    print(f"Normalized column '{col}' to range [0, 1]")
        
        return self
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to: {output_path}")
        return output_path
    
    def get_summary(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        summary = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': self.df.duplicated().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        
        return summary

def clean_csv_file(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    cleaner.load_data() \
           .handle_missing_values(strategy='mean') \
           .remove_duplicates() \
           .normalize_numeric()
    
    saved_path = cleaner.save_cleaned_data(output_file)
    
    summary = cleaner.get_summary()
    print("\nData Cleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return saved_path

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 5],
        'B': [10, 20, 30, np.nan, 50, 50],
        'C': ['x', 'y', 'z', 'x', 'y', 'y'],
        'D': [100, 200, 300, 400, 500, 500]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = "test_data.csv"
    test_df.to_csv(test_file, index=False)
    
    try:
        cleaned_file = clean_csv_file(test_file, "cleaned_test_data.csv")
        print(f"\nCleaning completed. Output file: {cleaned_file}")
    finally:
        import os
        if os.path.exists(test_file):
            os.remove(test_file)import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. 
            Can be 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            If None, missing values are not filled.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for required columns and minimum row count.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"