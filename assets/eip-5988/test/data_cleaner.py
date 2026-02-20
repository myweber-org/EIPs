
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
        
        if self.file_path.suffix == '.csv':
            self.df = pd.read_csv(self.file_path)
        elif self.file_path.suffix in ['.xlsx', '.xls']:
            self.df = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")
        
        print(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        return self.df
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        else:
            columns = [col for col in columns if col in self.df.columns]
        
        for column in columns:
            if self.df[column].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[column].mean()
                elif strategy == 'median':
                    fill_value = self.df[column].median()
                elif strategy == 'mode':
                    fill_value = self.df[column].mode()[0]
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[column])
                    continue
                else:
                    raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
                
                self.df[column].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{column}' using {strategy} strategy")
        
        return self.df
    
    def remove_duplicates(self, subset=None, keep='first'):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = initial_rows - len(self.df)
        
        if removed > 0:
            print(f"Removed {removed} duplicate rows")
        
        return self.df
    
    def normalize_columns(self, columns=None):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column in columns:
            if column in self.df.columns:
                min_val = self.df[column].min()
                max_val = self.df[column].max()
                
                if max_val > min_val:
                    self.df[f"{column}_normalized"] = (self.df[column] - min_val) / (max_val - min_val)
                    print(f"Normalized column '{column}' to range [0, 1]")
        
        return self.df
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            raise ValueError("No data to save. Load and clean data first.")
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        output_path = Path(output_path)
        
        if output_path.suffix == '.csv':
            self.df.to_csv(output_path, index=False)
        elif output_path.suffix in ['.xlsx', '.xls']:
            self.df.to_excel(output_path, index=False)
        else:
            raise ValueError("Unsupported output format. Use CSV or Excel.")
        
        print(f"Cleaned data saved to: {output_path}")
        return output_path
    
    def get_summary(self):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'data_types': self.df.dtypes.to_dict()
        }
        
        return summary

def clean_dataset(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    try:
        cleaner.load_data()
        cleaner.handle_missing_values(strategy='mean')
        cleaner.remove_duplicates()
        cleaner.normalize_columns()
        
        if output_file:
            cleaner.save_cleaned_data(output_file)
        else:
            cleaner.save_cleaned_data()
        
        summary = cleaner.get_summary()
        print(f"\nData cleaning completed:")
        print(f"  Rows: {summary['total_rows']}")
        print(f"  Columns: {summary['total_columns']}")
        print(f"  Missing values remaining: {summary['missing_values']}")
        print(f"  Duplicate rows remaining: {summary['duplicate_rows']}")
        
        return cleaner.df
        
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        clean_dataset(input_file, output_file)
    else:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        print("Example: python data_cleaner.py data.csv cleaned_data.csv")