
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.original_shape = data.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        clean_data = self.data.copy()
        for col in columns:
            Q1 = np.percentile(clean_data[col], 25)
            Q3 = np.percentile(clean_data[col], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)
            clean_data = clean_data[mask]
        
        removed_count = len(self.data) - len(clean_data)
        print(f"Removed {removed_count} outliers using IQR method")
        return clean_data
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        clean_data = self.data.copy()
        for col in columns:
            z_scores = np.abs(stats.zscore(clean_data[col]))
            mask = z_scores < threshold
            clean_data = clean_data[mask]
        
        removed_count = len(self.data) - len(clean_data)
        print(f"Removed {removed_count} outliers using Z-score method")
        return clean_data
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        normalized_data = self.data.copy()
        for col in columns:
            min_val = normalized_data[col].min()
            max_val = normalized_data[col].max()
            if max_val != min_val:
                normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        
        return normalized_data
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        normalized_data = self.data.copy()
        for col in columns:
            mean_val = normalized_data[col].mean()
            std_val = normalized_data[col].std()
            if std_val > 0:
                normalized_data[col] = (normalized_data[col] - mean_val) / std_val
        
        return normalized_data
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        cleaned_data = self.data.copy()
        for col in columns:
            if cleaned_data[col].isnull().any():
                if strategy == 'mean':
                    fill_value = cleaned_data[col].mean()
                elif strategy == 'median':
                    fill_value = cleaned_data[col].median()
                elif strategy == 'mode':
                    fill_value = cleaned_data[col].mode()[0]
                elif strategy == 'drop':
                    cleaned_data = cleaned_data.dropna(subset=[col])
                    continue
                else:
                    fill_value = 0
                
                cleaned_data[col] = cleaned_data[col].fillna(fill_value)
                print(f"Filled missing values in column {col} using {strategy} strategy")
        
        return cleaned_data
    
    def get_summary(self):
        summary = {
            'original_samples': self.original_shape[0],
            'original_features': self.original_shape[1],
            'current_samples': self.data.shape[0],
            'current_features': self.data.shape[1],
            'missing_values': self.data.isnull().sum().sum() if hasattr(self.data, 'isnull') else 0
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000)
    })
    
    data.iloc[10:15, 0] = np.nan
    data.iloc[20:25, 1] = np.nan
    data.iloc[5, 0] = 500
    data.iloc[8, 1] = 1000
    
    return data

if __name__ == "__main__":
    sample_data = create_sample_data()
    cleaner = DataCleaner(sample_data)
    
    print("Data Summary:")
    print(cleaner.get_summary())
    
    cleaned_data = cleaner.handle_missing_values(strategy='mean')
    cleaned_data = cleaner.remove_outliers_zscore(threshold=3)
    normalized_data = cleaner.normalize_minmax()
    
    print("\nCleaned Data Shape:", cleaned_data.shape)
    print("Normalized Data Statistics:")
    print(normalized_data.describe())
import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text by converting to lowercase, removing extra whitespace,
    and stripping special characters (except basic punctuation).
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s.,!?-]', '', x))
    return df

def remove_duplicate_rows(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def validate_email_column(df, column_name):
    """
    Validate email format in a specified column.
    Returns a boolean Series indicating valid emails.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[column_name].astype(str).str.match(email_pattern)

def main():
    # Example usage
    data = {
        'name': ['Alice', 'Bob', 'Alice', 'Charlie  '],
        'email': ['alice@example.com', 'invalid-email', 'alice@example.com', 'charlie@test.org'],
        'notes': ['Hello!', '  Multiple   spaces  ', 'Duplicate', 'Special#Chars']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean text columns
    df = clean_text_column(df, 'notes')
    print("After cleaning 'notes' column:")
    print(df)
    print("\n")
    
    # Remove duplicates
    df = remove_duplicate_rows(df, subset=['name', 'email'])
    print("After removing duplicates:")
    print(df)
    print("\n")
    
    # Validate emails
    df['valid_email'] = validate_email_column(df, 'email')
    print("With email validation:")
    print(df)

if __name__ == "__main__":
    main()
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
        keep: determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
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
        df: pandas DataFrame
        columns: list of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        Tuple of (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"