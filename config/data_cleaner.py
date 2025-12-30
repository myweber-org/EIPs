
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str or dict): Method to fill missing values ('mean', 'median', 'mode', or dict of column:value)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            for col, value in fill_missing.items():
                if col in cleaned_df.columns:
                    cleaned_df[col] = cleaned_df[col].fillna(value)
        elif fill_missing == 'mean':
            numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        elif fill_missing == 'median':
            numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                mode_val = cleaned_df[col].mode()
                if not mode_val.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} missing values remain in the dataset")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("Error: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True
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
        
        self.df = pd.read_csv(self.file_path)
        return self
        
    def remove_duplicates(self):
        if self.df is not None:
            self.df = self.df.drop_duplicates()
        return self
        
    def fill_missing_values(self, strategy='mean', fill_value=None):
        if self.df is not None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            if strategy == 'mean':
                self.df[numeric_cols] = self.df[numeric_cols].fillna(
                    self.df[numeric_cols].mean()
                )
            elif strategy == 'median':
                self.df[numeric_cols] = self.df[numeric_cols].fillna(
                    self.df[numeric_cols].median()
                )
            elif strategy == 'mode':
                self.df[numeric_cols] = self.df[numeric_cols].fillna(
                    self.df[numeric_cols].mode().iloc[0]
                )
            elif strategy == 'constant' and fill_value is not None:
                self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_value)
                
        return self
        
    def remove_outliers(self, method='iqr', threshold=1.5):
        if self.df is not None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if method == 'iqr':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    self.df = self.df[
                        (self.df[col] >= lower_bound) & 
                        (self.df[col] <= upper_bound)
                    ]
                    
        return self
        
    def standardize_columns(self):
        if self.df is not None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    self.df[col] = (self.df[col] - mean) / std
                    
        return self
        
    def save_cleaned_data(self, output_path):
        if self.df is not None:
            output_path = Path(output_path)
            self.df.to_csv(output_path, index=False)
            return output_path
        return None
        
    def get_summary(self):
        if self.df is not None:
            summary = {
                'original_rows': len(self.df),
                'columns': list(self.df.columns),
                'missing_values': self.df.isnull().sum().to_dict(),
                'data_types': self.df.dtypes.to_dict()
            }
            return summary
        return {}

def clean_csv_file(input_file, output_file, **kwargs):
    cleaner = DataCleaner(input_file)
    
    cleaner.load_data()
    
    if kwargs.get('remove_duplicates', True):
        cleaner.remove_duplicates()
        
    if kwargs.get('fill_missing', True):
        strategy = kwargs.get('fill_strategy', 'mean')
        fill_value = kwargs.get('fill_value')
        cleaner.fill_missing_values(strategy=strategy, fill_value=fill_value)
        
    if kwargs.get('remove_outliers', False):
        method = kwargs.get('outlier_method', 'iqr')
        threshold = kwargs.get('outlier_threshold', 1.5)
        cleaner.remove_outliers(method=method, threshold=threshold)
        
    if kwargs.get('standardize', False):
        cleaner.standardize_columns()
        
    output_path = cleaner.save_cleaned_data(output_file)
    summary = cleaner.get_summary()
    
    return output_path, summary
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = 'drop',
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and standardizing columns.
    
    Parameters:
    input_path: Path to input CSV file
    output_path: Path to save cleaned CSV file
    missing_strategy: Strategy for handling missing values ('drop', 'fill', 'mean')
    fill_value: Value to use when missing_strategy is 'fill'
    
    Returns:
    Cleaned DataFrame
    """
    try:
        df = pd.read_csv(input_path)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Handle missing values
        if missing_strategy == 'drop':
            df = df.dropna()
        elif missing_strategy == 'fill':
            if fill_value is not None:
                df = df.fillna(fill_value)
            else:
                df = df.fillna(0)
        elif missing_strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Convert column names to lowercase with underscores
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        
        print(f"Data cleaning completed. Cleaned data saved to {output_path}")
        print(f"Original shape: {df.shape}, Cleaned shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        raise
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        raise

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame for common data quality issues.
    
    Parameters:
    df: DataFrame to validate
    
    Returns:
    Boolean indicating if data passes validation
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if df.isnull().sum().sum() > len(df) * 0.5:
        print("Validation warning: More than 50% missing values")
        return False
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            if df[col].abs().max() > 1e10:
                print(f"Validation warning: Column {col} has extremely large values")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', None],
        'Age': [25, 30, None, 35],
        'Score': [85.5, 92.0, 78.5, 88.0]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data(
        input_path='sample_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='mean'
    )
    
    if validate_dataframe(cleaned_df):
        print("Data validation passed")
    else:
        print("Data validation failed")