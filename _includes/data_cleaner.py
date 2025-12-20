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
        return self.df
    
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
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                    continue
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                self.df[col] = self.df[col].fillna(fill_value)
                print(f"Filled missing values in '{col}' using {strategy} strategy")
        
        return self.df
    
    def remove_duplicates(self, subset=None, keep='first'):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed_count = initial_count - len(self.df)
        
        if removed_count > 0:
            print(f"Removed {removed_count} duplicate rows")
        
        return self.df
    
    def normalize_column(self, column, method='minmax'):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val != 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        print(f"Normalized column '{column}' using {method} method")
        return self.df
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            raise ValueError("No data to save. Load and clean data first.")
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to: {output_path}")
        return output_path

def process_csv_file(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    try:
        cleaner.load_data()
        cleaner.handle_missing_values(strategy='mean')
        cleaner.remove_duplicates()
        cleaner.save_cleaned_data(output_file)
        return True
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6],
        'value': [10.5, None, 15.2, 10.5, None, 20.1],
        'category': ['A', 'B', 'A', 'A', 'C', 'B']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = Path('test_data.csv')
    test_df.to_csv(test_file, index=False)
    
    process_csv_file('test_data.csv', 'cleaned_test_data.csv')
    
    if test_file.exists():
        test_file.unlink()