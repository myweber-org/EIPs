
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_dataimport csv
import re
from typing import List, Dict, Optional

def clean_string(value: str) -> str:
    """Remove extra whitespace and normalize string."""
    if not isinstance(value, str):
        return str(value)
    cleaned = re.sub(r'\s+', ' ', value.strip())
    return cleaned

def clean_numeric(value: str) -> Optional[float]:
    """Convert string to float, handling common issues."""
    if not value:
        return None
    cleaned = value.replace(',', '').replace('$', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return None

def validate_email(email: str) -> bool:
    """Basic email validation."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))

def read_csv_file(filepath: str) -> List[Dict]:
    """Read CSV file and return list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return data

def clean_csv_data(data: List[Dict]) -> List[Dict]:
    """Apply cleaning functions to CSV data."""
    cleaned_data = []
    for row in data:
        cleaned_row = {}
        for key, value in row.items():
            if key.lower().endswith('email'):
                cleaned_row[key] = clean_string(value) if validate_email(value) else None
            elif any(num_key in key.lower() for num_key in ['price', 'amount', 'quantity']):
                cleaned_row[key] = clean_numeric(value)
            else:
                cleaned_row[key] = clean_string(value)
        cleaned_data.append(cleaned_row)
    return cleaned_data

def write_csv_file(data: List[Dict], filepath: str) -> bool:
    """Write cleaned data to CSV file."""
    if not data:
        return False
    try:
        fieldnames = data[0].keys()
        with open(filepath, 'w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing CSV: {e}")
        return False

def process_csv(input_path: str, output_path: str) -> None:
    """Complete CSV processing pipeline."""
    print(f"Processing {input_path}...")
    raw_data = read_csv_file(input_path)
    if not raw_data:
        print("No data to process.")
        return
    cleaned_data = clean_csv_data(raw_data)
    success = write_csv_file(cleaned_data, output_path)
    if success:
        print(f"Cleaned data saved to {output_path}")
        print(f"Processed {len(cleaned_data)} records.")
    else:
        print("Failed to save cleaned data.")import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        return self
        
    def normalize_data(self, method='minmax', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_norm = self.df.copy()
        
        if method == 'minmax':
            for col in columns:
                if col in df_norm.columns:
                    min_val = df_norm[col].min()
                    max_val = df_norm[col].max()
                    if max_val != min_val:
                        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            for col in columns:
                if col in df_norm.columns:
                    mean_val = df_norm[col].mean()
                    std_val = df_norm[col].std()
                    if std_val > 0:
                        df_norm[col] = (df_norm[col] - mean_val) / std_val
        
        self.df = df_norm
        return self
        
    def handle_missing(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        
        for col in columns:
            if col in df_filled.columns and df_filled[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_filled[col].mean()
                elif strategy == 'median':
                    fill_value = df_filled[col].median()
                elif strategy == 'mode':
                    fill_value = df_filled[col].mode()[0]
                elif strategy == 'drop':
                    df_filled = df_filled.dropna(subset=[col])
                    continue
                else:
                    fill_value = 0
                
                df_filled[col] = df_filled[col].fillna(fill_value)
        
        self.df = df_filled
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]

def clean_dataset(df, outlier_columns=None, norm_method='minmax', missing_strategy='mean'):
    cleaner = DataCleaner(df)
    cleaner.handle_missing(strategy=missing_strategy)
    cleaner.remove_outliers_iqr(columns=outlier_columns)
    cleaner.normalize_data(method=norm_method)
    return cleaner.get_cleaned_data()import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                cleaned_df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value:.2f}")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col].fillna('Unknown', inplace=True)
            print(f"Filled missing values in categorical column '{col}' with 'Unknown'")
    
    print(f"Cleaning complete. Final dataset shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the dataset for required columns and minimum row count.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed. Missing columns: {missing_cols}")
            return False
    
    if len(df) < min_rows:
        print(f"Validation failed. Dataset has {len(df)} rows, minimum required is {min_rows}.")
        return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A'],
        'score': [85, 92, 92, 78, 88, np.nan]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='median')
    print("\nCleaned dataset:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['id', 'value', 'category'], min_rows=3)
    print(f"\nDataset is valid: {is_valid}")