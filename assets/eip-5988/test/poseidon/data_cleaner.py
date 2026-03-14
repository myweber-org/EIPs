
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (bool): Whether to fill missing values. Default is True.
    fill_value: Value to use for filling missing data. Default is 0.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def remove_outliers(df, column, threshold=3):
    """
    Remove outliers from a specific column using z-score method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to check for outliers.
    threshold (float): Z-score threshold. Default is 3.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    from scipy import stats
    import numpy as np
    
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    filtered_entries = z_scores < threshold
    return df[filtered_entries]

def normalize_column(df, column):
    """
    Normalize a column to range [0, 1] using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to normalize.
    
    Returns:
    pd.DataFrame: DataFrame with normalized column.
    """
    df_copy = df.copy()
    min_val = df_copy[column].min()
    max_val = df_copy[column].max()
    
    if max_val > min_val:
        df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    return df_copy

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, 6, 6, None, 9],
        'C': [10, 11, 11, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(file_path, output_path):
    try:
        df = pd.read_csv(file_path)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            original_count = len(df)
            df = remove_outliers_iqr(df, col)
            removed_count = original_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column '{col}'")
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    cleaned_data = clean_dataset(input_file, output_file)import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean.reset_index(drop=True)
        return removed_count
    
    def normalize_column(self, column, method='minmax'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Column '{column}' is not numeric")
        
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val == min_val:
                self.df[f'{column}_normalized'] = 0.5
            else:
                self.df[f'{column}_normalized'] = (self.df[column] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val == 0:
                self.df[f'{column}_normalized'] = 0
            else:
                self.df[f'{column}_normalized'] = (self.df[column] - mean_val) / std_val
        
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'")
        
        return self.df[f'{column}_normalized']
    
    def fill_missing_values(self, strategy='mean', custom_value=None):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'custom' and custom_value is not None:
                    fill_value = custom_value
                else:
                    continue
                
                self.df[col] = self.df[col].fillna(fill_value)
        
        return self.df.isnull().sum().sum()
    
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

def create_sample_data():
    np.random.seed(42)
    data = {
        'id': range(100),
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[10:15, 'feature_a'] = np.nan
    df.loc[95, 'feature_a'] = 500
    df.loc[96, 'feature_b'] = 1000
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Initial summary:")
    print(cleaner.get_summary())
    
    outliers_removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print(f"\nRemoved {outliers_removed} outliers")
    
    missing_filled = cleaner.fill_missing_values(strategy='mean')
    print(f"Filled {missing_filled} missing values")
    
    cleaner.normalize_column('feature_a', method='minmax')
    cleaner.normalize_column('feature_b', method='zscore')
    
    print("\nFinal summary:")
    print(cleaner.get_summary())
    
    cleaned_df = cleaner.get_cleaned_data()
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())
import pandas as pd
import numpy as np
from typing import List, Optional

def clean_dataframe(df: pd.DataFrame, 
                    drop_duplicates: bool = True,
                    fill_missing: Optional[str] = 'mean',
                    columns_to_standardize: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Clean a pandas DataFrame by removing duplicates, handling missing values,
    and standardizing specified columns.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    
    if columns_to_standardize:
        for col in columns_to_standardize:
            if col in cleaned_df.columns and cleaned_df[col].dtype in [np.float64, np.int64]:
                mean = cleaned_df[col].mean()
                std = cleaned_df[col].std()
                if std > 0:
                    cleaned_df[col] = (cleaned_df[col] - mean) / std
    
    return cleaned_df

def validate_dataframe(df: pd.DataFrame, 
                       required_columns: List[str],
                       min_rows: int = 1) -> bool:
    """
    Validate that a DataFrame meets basic requirements.
    """
    if len(df) < min_rows:
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, None, 15.3, 20.1, None],
        'category': ['A', 'B', 'B', 'A', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataframe(df, 
                              drop_duplicates=True,
                              fill_missing='mean',
                              columns_to_standardize=['value'])
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, required_columns=['id', 'value'])
    print(f"\nDataFrame is valid: {is_valid}")import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.random.normal(300, 50, 5)
        ])
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original summary statistics:")
    original_stats = calculate_summary_stats(df, 'values')
    for key, value in original_stats.items():
        print(f"  {key}: {value:.2f}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary statistics:")
    cleaned_stats = calculate_summary_stats(cleaned_df, 'values')
    for key, value in cleaned_stats.items():
        print(f"  {key}: {value:.2f}")
    
    removed_count = len(df) - len(cleaned_df)
    print(f"\nRemoved {removed_count} outliers")

if __name__ == "__main__":
    example_usage()