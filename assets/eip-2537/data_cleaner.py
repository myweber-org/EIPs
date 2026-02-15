def clean_data(data):
    unique_data = list(set(data))
    sorted_data = sorted(unique_data)
    return sorted_data

def process_numbers(numbers):
    cleaned = clean_data(numbers)
    total = sum(cleaned)
    average = total / len(cleaned) if cleaned else 0
    return {
        'cleaned_data': cleaned,
        'total': total,
        'average': average
    }

if __name__ == "__main__":
    sample_data = [5, 2, 8, 2, 5, 9, 1, 8]
    result = process_numbers(sample_data)
    print(f"Cleaned data: {result['cleaned_data']}")
    print(f"Total: {result['total']}")
    print(f"Average: {result['average']}")
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
        
        print(f"Loaded data with shape: {self.df.shape}")
        return self
    
    def remove_duplicates(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_rows - len(self.df)
        print(f"Removed {removed} duplicate rows")
        return self
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
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
                    print(f"Dropped rows with missing values in column: {column}")
                    continue
                else:
                    fill_value = strategy
                
                self.df[column] = self.df[column].fillna(fill_value)
                print(f"Filled missing values in '{column}' using {strategy} strategy")
        
        return self
    
    def remove_outliers(self, columns=None, method='iqr', threshold=1.5):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        initial_rows = len(self.df)
        
        for column in columns:
            if method == 'iqr':
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
                self.df = self.df[mask]
            
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(self.df[column]))
                self.df = self.df[z_scores < threshold]
        
        removed = initial_rows - len(self.df)
        print(f"Removed {removed} outliers")
        return self
    
    def standardize_columns(self, columns=None):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column in columns:
            mean = self.df[column].mean()
            std = self.df[column].std()
            if std > 0:
                self.df[f"{column}_standardized"] = (self.df[column] - mean) / std
        
        return self
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            raise ValueError("No data to save. Perform cleaning operations first.")
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        output_path = Path(output_path)
        
        if output_path.suffix == '.csv':
            self.df.to_csv(output_path, index=False)
        elif output_path.suffix in ['.xlsx', '.xls']:
            self.df.to_excel(output_path, index=False)
        else:
            output_path = output_path.with_suffix('.csv')
            self.df.to_csv(output_path, index=False)
        
        print(f"Cleaned data saved to: {output_path}")
        return output_path
    
    def get_summary(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        summary = {
            'original_file': str(self.file_path),
            'current_shape': self.df.shape,
            'columns': list(self.df.columns),
            'data_types': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_stats': {}
        }
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary['numeric_stats'][col] = {
                'mean': self.df[col].mean(),
                'median': self.df[col].median(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max()
            }
        
        return summary

def clean_csv_file(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    try:
        cleaner.load_data()
        cleaner.remove_duplicates()
        cleaner.handle_missing_values(strategy='median')
        cleaner.remove_outliers(threshold=1.5)
        cleaner.standardize_columns()
        
        if output_file:
            output_path = cleaner.save_cleaned_data(output_file)
        else:
            output_path = cleaner.save_cleaned_data()
        
        summary = cleaner.get_summary()
        print(f"\nCleaning completed successfully!")
        print(f"Output file: {output_path}")
        print(f"Final data shape: {summary['current_shape']}")
        
        return cleaner.df
        
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        clean_csv_file(input_file, output_file)
    else:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        print("Example: python data_cleaner.py data.csv cleaned_data.csv")