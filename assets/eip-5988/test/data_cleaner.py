
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
        
        return self.df.head()
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            self.load_data()
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column in columns:
            if column in self.df.columns:
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
                
                self.df[column] = self.df[column].fillna(fill_value)
        
        return self.df
    
    def remove_duplicates(self, subset=None, keep='first'):
        if self.df is None:
            self.load_data()
        
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed_count = initial_count - len(self.df)
        
        return self.df, removed_count
    
    def normalize_columns(self, columns=None):
        if self.df is None:
            self.load_data()
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column in columns:
            if column in self.df.columns:
                min_val = self.df[column].min()
                max_val = self.df[column].max()
                
                if max_val != min_val:
                    self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        
        return self.df
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            raise ValueError("No data to save. Please load and clean data first.")
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        output_path = Path(output_path)
        
        if output_path.suffix == '.csv':
            self.df.to_csv(output_path, index=False)
        elif output_path.suffix in ['.xlsx', '.xls']:
            self.df.to_excel(output_path, index=False)
        else:
            raise ValueError("Unsupported output format. Use CSV or Excel files.")
        
        return output_path
    
    def get_summary(self):
        if self.df is None:
            self.load_data()
        
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'data_types': self.df.dtypes.to_dict(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        
        return summary

def clean_dataset(input_file, output_file=None, missing_strategy='mean'):
    cleaner = DataCleaner(input_file)
    cleaner.load_data()
    cleaner.handle_missing_values(strategy=missing_strategy)
    cleaner.remove_duplicates()
    cleaner.normalize_columns()
    
    if output_file:
        saved_path = cleaner.save_cleaned_data(output_file)
    else:
        saved_path = cleaner.save_cleaned_data()
    
    summary = cleaner.get_summary()
    
    return cleaner.df, saved_path, summary