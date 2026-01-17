import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.df = None
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"Loaded {len(self.df)} rows from {self.filepath.name}")
            return True
        except FileNotFoundError:
            print(f"Error: File {self.filepath} not found")
            return False
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def remove_duplicates(self):
        if self.df is not None:
            initial_count = len(self.df)
            self.df.drop_duplicates(inplace=True)
            removed = initial_count - len(self.df)
            print(f"Removed {removed} duplicate rows")
            return removed
        return 0
    
    def fill_missing_values(self, strategy='mean', columns=None):
        if self.df is not None:
            if columns is None:
                columns = self.df.select_dtypes(include=[np.number]).columns
            
            for col in columns:
                if col in self.df.columns:
                    if strategy == 'mean':
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    elif strategy == 'median':
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    elif strategy == 'mode':
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    elif strategy == 'zero':
                        self.df[col].fillna(0, inplace=True)
            
            print(f"Filled missing values using {strategy} strategy")
            return True
        return False
    
    def remove_outliers(self, columns=None, method='iqr', threshold=1.5):
        if self.df is not None:
            if columns is None:
                columns = self.df.select_dtypes(include=[np.number]).columns
            
            initial_count = len(self.df)
            
            for col in columns:
                if col in self.df.columns:
                    if method == 'iqr':
                        Q1 = self.df[col].quantile(0.25)
                        Q3 = self.df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            
            removed = initial_count - len(self.df)
            print(f"Removed {removed} outliers using {method} method")
            return removed
        return 0
    
    def standardize_columns(self, columns=None):
        if self.df is not None:
            if columns is None:
                columns = self.df.select_dtypes(include=[np.number]).columns
            
            for col in columns:
                if col in self.df.columns:
                    mean = self.df[col].mean()
                    std = self.df[col].std()
                    if std > 0:
                        self.df[col] = (self.df[col] - mean) / std
            
            print(f"Standardized {len(columns)} columns")
            return True
        return False
    
    def save_cleaned_data(self, output_path=None):
        if self.df is not None:
            if output_path is None:
                output_path = self.filepath.parent / f"cleaned_{self.filepath.name}"
            
            self.df.to_csv(output_path, index=False)
            print(f"Saved cleaned data to {output_path}")
            return output_path
        return None
    
    def get_summary(self):
        if self.df is not None:
            summary = {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'missing_values': self.df.isnull().sum().sum(),
                'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
            }
            return summary
        return {}

def clean_csv_file(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    if cleaner.load_data():
        cleaner.remove_duplicates()
        cleaner.fill_missing_values(strategy='mean')
        cleaner.remove_outliers()
        cleaner.standardize_columns()
        
        output_path = cleaner.save_cleaned_data(output_file)
        summary = cleaner.get_summary()
        
        print("\nData Cleaning Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        return output_path
    return None

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'Category': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = Path("test_data.csv")
    test_df.to_csv(test_file, index=False)
    
    cleaned_file = clean_csv_file(test_file)
    
    if cleaned_file:
        print(f"\nCleaned file saved to: {cleaned_file}")
    
    test_file.unlink()
    if cleaned_file and Path(cleaned_file).exists():
        Path(cleaned_file).unlink()