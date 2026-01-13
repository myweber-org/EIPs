import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Loaded {len(self.df)} rows from {self.file_path.name}")
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                missing_count = self.df[col].isnull().sum()
                if missing_count > 0:
                    if strategy == 'mean':
                        fill_value = self.df[col].mean()
                    elif strategy == 'median':
                        fill_value = self.df[col].median()
                    elif strategy == 'mode':
                        fill_value = self.df[col].mode()[0]
                    else:
                        fill_value = 0
                    
                    self.df[col].fillna(fill_value, inplace=True)
                    print(f"Filled {missing_count} missing values in '{col}' with {strategy} value: {fill_value:.2f}")
    
    def remove_duplicates(self, subset=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        initial_count = len(self.df)
        self.df.drop_duplicates(subset=subset, inplace=True, keep='first')
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
    
    def normalize_column(self, column_name):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        if column_name in self.df.columns:
            col_min = self.df[column_name].min()
            col_max = self.df[column_name].max()
            
            if col_max != col_min:
                self.df[column_name] = (self.df[column_name] - col_min) / (col_max - col_min)
                print(f"Normalized column '{column_name}' to range [0, 1]")
            else:
                print(f"Column '{column_name}' has constant values, skipping normalization")
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")
        return output_path
    
    def get_summary(self):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
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

def clean_csv_file(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    if cleaner.load_data():
        print("Starting data cleaning process...")
        
        cleaner.handle_missing_values(strategy='mean')
        cleaner.remove_duplicates()
        
        summary = cleaner.get_summary()
        print(f"\nData Summary:")
        print(f"Rows: {summary['total_rows']}")
        print(f"Columns: {summary['total_columns']}")
        print(f"Numeric columns: {len(summary['numeric_columns'])}")
        print(f"Categorical columns: {len(summary['categorical_columns'])}")
        
        saved_path = cleaner.save_cleaned_data(output_file)
        return saved_path
    
    return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        clean_csv_file(input_file, output_file)
    else:
        print("Usage: python data_cleaner.py <input_csv> [output_csv]")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, outlier_method='iqr', normalize_method='minmax', columns=None):
    """
    Main function to clean dataset by removing outliers and normalizing
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in columns:
        if column not in data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        if normalize_method == 'minmax':
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[column] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate summary statistics for the dataset
    """
    summary = {
        'original_rows': len(data),
        'original_columns': len(data.columns),
        'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(data.select_dtypes(include=['object']).columns),
        'missing_values': data.isnull().sum().sum(),
        'duplicate_rows': data.duplicated().sum()
    }
    
    numeric_summary = data.describe().to_dict() if not data.select_dtypes(include=[np.number]).empty else {}
    
    return {
        'dataset_summary': summary,
        'numeric_statistics': numeric_summary
    }