
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0):
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
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.fillna(fill_value)
        missing_after = cleaned_df.isnull().sum().sum()
        print(f"Filled {missing_before} missing values")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['missing_columns'] = missing_columns
    
    return validation_results

def process_csv_file(input_path, output_path=None, **clean_kwargs):
    """
    Process a CSV file through the data cleaning pipeline.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save cleaned CSV file
        **clean_kwargs: Additional arguments for clean_dataframe
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded data from {input_path}")
        print(f"Original shape: {df.shape}")
        
        cleaned_df = clean_dataframe(df, **clean_kwargs)
        print(f"Cleaned shape: {cleaned_df.shape}")
        
        validation = validate_dataframe(cleaned_df)
        print(f"Validation results: {validation}")
        
        if output_path:
            cleaned_df.to_csv(output_path, index=False)
            print(f"Saved cleaned data to {output_path}")
        
        return cleaned_df
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10, 20, 20, np.nan, 40, 50],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Sample data created")
    print(df)
    
    cleaned = clean_dataframe(df, fill_value='unknown')
    print("\nCleaned data:")
    print(cleaned)
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0):
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
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.fillna(fill_value)
        missing_after = cleaned_df.isnull().sum().sum()
        print(f"Filled {missing_before - missing_after} missing values")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def process_dataset(file_path, output_path=None):
    """
    Load, clean, and optionally save a dataset.
    
    Args:
        file_path (str): Path to input CSV file
        output_path (str, optional): Path to save cleaned data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        if validate_dataframe(df):
            cleaned_df = clean_dataframe(df)
            
            if output_path:
                cleaned_df.to_csv(output_path, index=False)
                print(f"Cleaned data saved to {output_path}")
            
            return cleaned_df
        else:
            print("Data validation failed")
            return None
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return None

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10, 20, 20, None, 40, 50],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataframe(df)
    print(cleaned)
import pandas as pd
import re

def clean_dataset(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing
    specified string columns (strip whitespace, lowercase).
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_count = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    duplicates_removed = initial_count - len(cleaned_df)
    
    # Normalize specified string columns
    for col in column_names:
        if col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].apply(
                    lambda x: re.sub(r'\s+', ' ', str(x).strip().lower()) 
                    if pd.notnull(x) else x
                )
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    print(f"Removed {duplicates_removed} duplicate rows")
    print(f"Normalized columns: {column_names}")
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column using regex pattern.
    Returns DataFrame with validation results.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    df['email_valid'] = df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    valid_count = df['email_valid'].sum()
    total_count = len(df)
    
    print(f"Email validation results:")
    print(f"Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    
    return dfimport pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, output_path):
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_minmax(df, col)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Original shape: {pd.read_csv(file_path).shape}, Cleaned shape: {df.shape}")

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_missing(self, threshold=0.3):
        missing_percent = self.df.isnull().sum() / len(self.df)
        columns_to_drop = missing_percent[missing_percent > threshold].index
        self.df = self.df.drop(columns=columns_to_drop)
        return self
    
    def fill_numeric_missing(self, method='median'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'median':
            fill_values = self.df[numeric_cols].median()
        elif method == 'mean':
            fill_values = self.df[numeric_cols].mean()
        else:
            fill_values = 0
            
        self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_values)
        return self
    
    def detect_outliers_zscore(self, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        z_scores = np.abs(stats.zscore(self.df[numeric_cols]))
        outlier_mask = (z_scores > threshold).any(axis=1)
        return outlier_mask
    
    def remove_outliers(self, threshold=3):
        outlier_mask = self.detect_outliers_zscore(threshold)
        self.df = self.df[~outlier_mask]
        return self
    
    def normalize_data(self, method='minmax'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            min_vals = self.df[numeric_cols].min()
            max_vals = self.df[numeric_cols].max()
            self.df[numeric_cols] = (self.df[numeric_cols] - min_vals) / (max_vals - min_vals)
        elif method == 'zscore':
            mean_vals = self.df[numeric_cols].mean()
            std_vals = self.df[numeric_cols].std()
            self.df[numeric_cols] = (self.df[numeric_cols] - mean_vals) / std_vals
            
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        cleaned_shape = self.df.shape
        rows_removed = self.original_shape[0] - cleaned_shape[0]
        cols_removed = self.original_shape[1] - cleaned_shape[1]
        
        summary = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': cleaned_shape[0],
            'cleaned_columns': cleaned_shape[1],
            'rows_removed': rows_removed,
            'columns_removed': cols_removed,
            'missing_values': self.df.isnull().sum().sum()
        }
        
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(100, 10), 'feature_a'] = np.nan
    df.loc[np.random.choice(100, 5), 'feature_b'] = np.nan
    
    outlier_indices = np.random.choice(100, 3)
    df.loc[outlier_indices, 'feature_a'] = 500
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    
    cleaner = DataCleaner(sample_df)
    cleaned_df = (cleaner
                 .remove_missing(threshold=0.2)
                 .fill_numeric_missing('median')
                 .remove_outliers(threshold=3)
                 .normalize_data('minmax')
                 .get_cleaned_data())
    
    summary = cleaner.get_summary()
    print("\nCleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())