
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
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_clean (list, optional): List of column names to apply string normalization.
            If None, all object dtype columns are cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Determine columns to normalize
    if columns_to_clean is None:
        columns_to_clean = df_cleaned.select_dtypes(include=['object']).columns.tolist()
    
    # Apply string normalization to specified columns
    for col in columns_to_clean:
        if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].apply(_normalize_string)
    
    return df_cleaned

def _normalize_string(s):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Args:
        s (str): Input string.
    
    Returns:
        str: Normalized string.
    """
    if not isinstance(s, str):
        return s
    
    # Convert to lowercase
    s = s.lower()
    
    # Remove extra whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    
    # Remove special characters except alphanumeric and spaces
    s = re.sub(r'[^\w\s]', '', s)
    
    return s

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with an additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    df = df.copy()
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    return df