
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
                            Options: 'mean', 'median', 'drop'.
    outlier_threshold (float): Number of standard deviations for outlier detection.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    else:
        raise ValueError("Invalid missing_strategy. Choose from 'mean', 'median', or 'drop'.")
    
    # Remove outliers using z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
        cleaned_df = cleaned_df[z_scores < outlier_threshold]
    
    # Reset index after removing outliers
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that required columns exist in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if all required columns exist, False otherwise.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 8],
        'C': [9, 10, 11, 12, 13]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    # Clean the data
    cleaned = clean_dataset(df, missing_strategy='mean', outlier_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate required columns
    required = ['A', 'B', 'C']
    is_valid = validate_data(cleaned, required)
    print(f"\nData validation result: {is_valid}")import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    If columns specified, only check those columns.
    """
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()

def fill_missing_with_mean(df, columns):
    """
    Fill missing values in specified columns with column mean.
    """
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            df_filled[col] = df_filled[col].fillna(mean_val)
    return df_filled

def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method for a specific column.
    Returns boolean Series indicating outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def remove_outliers(df, column):
    """
    Remove rows where specified column contains outliers (IQR method).
    """
    outliers = detect_outliers_iqr(df, column)
    return df[~outliers]

def standardize_column(df, column):
    """
    Standardize a column to have mean=0 and std=1.
    """
    if column in df.columns:
        df_standardized = df.copy()
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            df_standardized[column] = (df[column] - mean_val) / std_val
        return df_standardized
    return df

def clean_dataset(df, missing_strategy='remove', outlier_columns=None):
    """
    Main cleaning function with configurable strategies.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'remove':
        cleaned_df = remove_missing_rows(cleaned_df)
    elif missing_strategy == 'mean':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df = fill_missing_with_mean(cleaned_df, numeric_cols)
    
    # Handle outliers
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers(cleaned_df, col)
    
    return cleaned_df
import numpy as np
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
        removed_count = self.original_shape[0] - self.df.shape[0]
        return removed_count
    
    def normalize_data(self, method='minmax', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        
        if method == 'minmax':
            for col in columns:
                if col in df_normalized.columns:
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()
                    if max_val > min_val:
                        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            for col in columns:
                if col in df_normalized.columns:
                    mean_val = df_normalized[col].mean()
                    std_val = df_normalized[col].std()
                    if std_val > 0:
                        df_normalized[col] = (df_normalized[col] - mean_val) / std_val
        
        elif method == 'robust':
            for col in columns:
                if col in df_normalized.columns:
                    median_val = df_normalized[col].median()
                    iqr_val = stats.iqr(df_normalized[col])
                    if iqr_val > 0:
                        df_normalized[col] = (df_normalized[col] - median_val) / iqr_val
        
        self.df = df_normalized
        return self.df
    
    def handle_missing_values(self, strategy='mean', columns=None):
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
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    continue
                
                df_filled[col] = df_filled[col].fillna(fill_value)
        
        self.df = df_filled
        missing_filled = self.df.isnull().sum().sum()
        return missing_filled
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        return summary

def clean_dataset(df, remove_outliers=True, normalize=True, handle_missing=True):
    cleaner = DataCleaner(df)
    
    if handle_missing:
        cleaner.handle_missing_values(strategy='mean')
    
    if remove_outliers:
        cleaner.remove_outliers_iqr()
    
    if normalize:
        cleaner.normalize_data(method='minmax')
    
    return cleaner.get_cleaned_data(), cleaner.get_summary()
import pandas as pd
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

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['X', 'Y', 'Z'], 200)
    })
    
    numeric_cols = ['feature_a', 'feature_b']
    result = clean_dataset(sample_data, numeric_cols)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {result.shape}")
    print(result.head())import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None):
    """
    Load a CSV file, perform basic cleaning operations,
    and optionally save the cleaned data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df)
        print(f"Removed {duplicates_removed} duplicate rows")
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Filled missing values in '{col}' with median: {median_val}")
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"Filled missing values in '{col}' with mode: {mode_val}")
        
        # Remove columns with more than 50% missing values
        missing_threshold = 0.5
        cols_to_drop = []
        for col in df.columns:
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > missing_threshold:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            print(f"Dropped columns with >50% missing values: {cols_to_drop}")
        
        print(f"Cleaned data shape: {df.shape}")
        
        # Save cleaned data if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    # Check for remaining missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print("Warning: DataFrame still contains missing values")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  {col}: {count} missing values")
    
    return True

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df)
        if is_valid:
            print("Data cleaning completed successfully")
        else:
            print("Data validation failed")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    original_shape = df.shape
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {original_shape[0] - cleaned_df.shape[0]} duplicate rows")
    
    # Handle missing values
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Removed rows with missing values")
        else:
            for column in cleaned_df.select_dtypes(include=[np.number]).columns:
                if cleaned_df[column].isnull().any():
                    if fill_missing == 'mean':
                        fill_value = cleaned_df[column].mean()
                    elif fill_missing == 'median':
                        fill_value = cleaned_df[column].median()
                    elif fill_missing == 'mode':
                        fill_value = cleaned_df[column].mode()[0]
                    else:
                        raise ValueError(f"Unknown fill method: {fill_missing}")
                    
                    cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                    print(f"Filled missing values in column '{column}' with {fill_missing}: {fill_value}")
    
    # Report cleaning results
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Rows removed: {original_shape[0] - cleaned_df.shape[0]}")
    print(f"Columns removed: {original_shape[1] - cleaned_df.shape[1]}")
    
    return cleaned_df

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate the structure and content of a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
        numeric_columns (list): List of column names that should be numeric
    
    Returns:
        dict: Dictionary containing validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check numeric columns
    if numeric_columns:
        for column in numeric_columns:
            if column in df.columns:
                if not pd.api.types.is_numeric_dtype(df[column]):
                    validation_results['warnings'].append(f"Column '{column}' is not numeric")
    
    # Check for infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        infinite_count = np.isinf(numeric_df).sum().sum()
        if infinite_count > 0:
            validation_results['warnings'].append(f"Found {infinite_count} infinite values")
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data with issues
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10.5, 20.3, np.nan, 30.1, 40.7, np.nan],
        'category': ['A', 'B', 'A', 'A', 'C', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    
    print("\n" + "="*50 + "\n")
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    print("\n" + "="*50 + "\n")
    validation = validate_data(
        cleaned_df,
        required_columns=['id', 'value', 'category'],
        numeric_columns=['id', 'value']
    )
    
    print("Validation Results:")
    print(f"Is valid: {validation['is_valid']}")
    if validation['errors']:
        print("Errors:", validation['errors'])
    if validation['warnings']:
        print("Warnings:", validation['warnings'])
import pandas as pd
import numpy as np
from typing import Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
    
    def remove_duplicates(self) -> 'DataCleaner':
        self.df = self.df.drop_duplicates()
        return self
    
    def fill_missing_numeric(self, strategy: str = 'mean', fill_value: Optional[float] = None) -> 'DataCleaner':
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean' and fill_value is None:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median' and fill_value is None:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif fill_value is not None:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_value)
        
        return self
    
    def drop_columns_missing_threshold(self, threshold: float = 0.5) -> 'DataCleaner':
        missing_ratio = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index
        self.df = self.df.drop(columns=cols_to_drop)
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
    
    def get_cleaning_report(self) -> dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1]
        }

def clean_csv_file(input_path: str, output_path: str, **kwargs) -> dict:
    df = pd.read_csv(input_path)
    cleaner = DataCleaner(df)
    
    cleaner.remove_duplicates()
    
    if 'numeric_strategy' in kwargs:
        cleaner.fill_missing_numeric(strategy=kwargs['numeric_strategy'])
    else:
        cleaner.fill_missing_numeric()
    
    if 'missing_threshold' in kwargs:
        cleaner.drop_columns_missing_threshold(kwargs['missing_threshold'])
    
    cleaned_df = cleaner.get_cleaned_data()
    cleaned_df.to_csv(output_path, index=False)
    
    return cleaner.get_cleaning_report()