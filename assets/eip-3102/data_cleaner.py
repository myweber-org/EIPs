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
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Column names to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
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

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David', 'Eve'],
        'score': ['85', '90', '90', '95', '100', '100', '88']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_duplicates(df, subset=['id', 'name'])
    print("After removing duplicates:")
    print(cleaned_df)
    print()
    
    cleaned_df = clean_numeric_columns(cleaned_df, ['score'])
    print("After cleaning numeric columns:")
    print(cleaned_df)
    print()
    
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'score'])
    print(f"Validation result: {is_valid}")
    print(f"Message: {message}")

if __name__ == "__main__":
    main()
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
                        print(f"Dropped {missing_count} rows with missing values in column '{col}'")
                        continue
                    else:
                        fill_value = strategy
                    
                    self.df[col] = self.df[col].fillna(fill_value)
                    print(f"Filled {missing_count} missing values in column '{col}' with {fill_value}")
        
        return self
    
    def remove_outliers(self, columns=None, method='iqr', threshold=1.5):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        initial_rows = len(self.df)
        
        for col in columns:
            if col in self.df.columns:
                if method == 'iqr':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                    self.df = self.df[mask]
        
        removed = initial_rows - len(self.df)
        print(f"Removed {removed} outlier rows")
        return self
    
    def standardize_columns(self, columns=None):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    self.df[col] = (self.df[col] - mean) / std
                    print(f"Standardized column '{col}'")
        
        return self
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
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
        
        print(f"Saved cleaned data to {output_path}")
        return output_path
    
    def get_summary(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': self.df.duplicated().sum()
        }
        
        return summary

def clean_dataset(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    cleaner.load_data() \
           .remove_duplicates() \
           .handle_missing_values(strategy='mean') \
           .remove_outliers() \
           .standardize_columns()
    
    output_path = cleaner.save_cleaned_data(output_file)
    summary = cleaner.get_summary()
    
    print("\nData Cleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return cleaner.df, output_path