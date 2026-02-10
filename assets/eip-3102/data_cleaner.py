import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.data = None
        
    def load_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if self.file_path.suffix == '.csv':
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.suffix in ['.xlsx', '.xls']:
            self.data = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format")
        
        print(f"Loaded {len(self.data)} rows and {len(self.data.columns)} columns")
        return self
    
    def remove_duplicates(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first")
        
        initial_count = len(self.data)
        self.data = self.data.drop_duplicates()
        removed = initial_count - len(self.data)
        print(f"Removed {removed} duplicate rows")
        return self
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first")
        
        if columns is None:
            columns = self.data.columns
        
        for column in columns:
            if column in self.data.columns and self.data[column].isnull().any():
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.data[column]):
                    fill_value = self.data[column].mean()
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.data[column]):
                    fill_value = self.data[column].median()
                elif strategy == 'mode':
                    fill_value = self.data[column].mode()[0]
                elif strategy == 'drop':
                    self.data = self.data.dropna(subset=[column])
                    continue
                else:
                    fill_value = 0
                
                self.data[column] = self.data[column].fillna(fill_value)
                print(f"Filled missing values in '{column}' using {strategy} strategy")
        
        return self
    
    def normalize_numeric_columns(self, columns=None):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first")
        
        if columns is None:
            columns = [col for col in self.data.columns 
                      if pd.api.types.is_numeric_dtype(self.data[col])]
        
        for column in columns:
            if column in self.data.columns and pd.api.types.is_numeric_dtype(self.data[column]):
                min_val = self.data[column].min()
                max_val = self.data[column].max()
                
                if max_val > min_val:
                    self.data[column] = (self.data[column] - min_val) / (max_val - min_val)
                    print(f"Normalized column '{column}' to range [0, 1]")
        
        return self
    
    def save_cleaned_data(self, output_path=None):
        if self.data is None:
            raise ValueError("No data to save")
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        if output_path.suffix == '.csv':
            self.data.to_csv(output_path, index=False)
        elif output_path.suffix in ['.xlsx', '.xls']:
            self.data.to_excel(output_path, index=False)
        
        print(f"Saved cleaned data to: {output_path}")
        return output_path
    
    def get_summary(self):
        if self.data is None:
            return {}
        
        summary = {
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'missing_values': self.data.isnull().sum().sum(),
            'numeric_columns': len([col for col in self.data.columns 
                                   if pd.api.types.is_numeric_dtype(self.data[col])]),
            'categorical_columns': len([col for col in self.data.columns 
                                       if pd.api.types.is_string_dtype(self.data[col])])
        }
        
        return summary

def clean_dataset(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    try:
        cleaner.load_data()
        cleaner.remove_duplicates()
        cleaner.handle_missing_values(strategy='mean')
        cleaner.normalize_numeric_columns()
        
        if output_file:
            output_path = cleaner.save_cleaned_data(output_file)
        else:
            output_path = cleaner.save_cleaned_data()
        
        summary = cleaner.get_summary()
        print(f"Cleaning complete. Summary: {summary}")
        
        return cleaner.data, output_path
        
    except Exception as e:
        print(f"Error during cleaning: {e}")
        raise

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10.5, 20.3, np.nan, 40.7, 50.1, 50.1, 60.9],
        'category': ['A', 'B', 'A', np.nan, 'B', 'B', 'A'],
        'score': [85, 92, 78, 88, np.nan, 88, 95]
    }
    
    df = pd.DataFrame(sample_data)
    test_file = Path("test_data.csv")
    df.to_csv(test_file, index=False)
    
    cleaned_data, output_path = clean_dataset(test_file)
    print(f"Output saved to: {output_path}")
    
    test_file.unlink()
    Path(output_path).unlink()