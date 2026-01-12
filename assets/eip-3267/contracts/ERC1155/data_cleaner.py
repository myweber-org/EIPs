
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (bool): Whether to fill missing values
        fill_value: Value to use for filling missing data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
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
            raise ValueError("Unsupported file format")
        
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
                    elif strategy == 'drop':
                        self.df = self.df.dropna(subset=[col])
                        print(f"Dropped rows with missing values in column: {col}")
                        continue
                    else:
                        fill_value = strategy
                    
                    self.df[col] = self.df[col].fillna(fill_value)
                    print(f"Filled {missing_count} missing values in column '{col}' with {fill_value}")
        
        return self
    
    def normalize_columns(self, columns=None):
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
                    print(f"Normalized column: {col}")
        
        return self
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            raise ValueError("No data to save. Process data first.")
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        output_path = Path(output_path)
        
        if output_path.suffix == '.csv':
            self.df.to_csv(output_path, index=False)
        elif output_path.suffix in ['.xlsx', '.xls']:
            self.df.to_excel(output_path, index=False)
        
        print(f"Saved cleaned data to: {output_path}")
        return output_path
    
    def get_summary(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        summary = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'data_types': self.df.dtypes.to_dict()
        }
        
        return summary

def clean_csv_file(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    try:
        cleaner.load_data()
        cleaner.remove_duplicates()
        cleaner.handle_missing_values(strategy='mean')
        cleaner.normalize_columns()
        
        if output_file:
            output_path = cleaner.save_cleaned_data(output_file)
        else:
            output_path = cleaner.save_cleaned_data()
        
        summary = cleaner.get_summary()
        print(f"Data cleaning completed. Summary: {summary}")
        
        return output_path, summary
        
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        raise

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, 8.7, 8.7],
        'category': ['A', 'B', 'A', None, 'C', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    try:
        output_file, summary = clean_csv_file('sample_data.csv')
        print(f"Output saved to: {output_file}")
        print(f"Final summary: {summary}")
    finally:
        import os
        if os.path.exists('sample_data.csv'):
            os.remove('sample_data.csv')
        if os.path.exists('cleaned_sample_data.csv'):
            os.remove('cleaned_sample_data.csv')
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Original dataset shape: {data.shape}")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            original_count = len(data)
            data = remove_outliers_iqr(data, column)
            removed_count = original_count - len(data)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column: {column}")
        
        print(f"Cleaned dataset shape: {data.shape}")
        return data
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    cleaned_data = clean_dataset("sample_data.csv")
    if cleaned_data is not None:
        cleaned_data.to_csv("cleaned_data.csv", index=False)
        print("Cleaned data saved to cleaned_data.csv")