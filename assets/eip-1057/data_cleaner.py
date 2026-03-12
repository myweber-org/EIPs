
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def main():
    data = {'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 15, 10, 9, 200]}
    df = pd.DataFrame(data)
    print("Original data:")
    print(df)
    
    cleaned_df = clean_dataset(df, ['values'])
    print("\nCleaned data:")
    print(cleaned_df)
    
    print(f"\nRemoved {len(df) - len(cleaned_df)} outliers")

if __name__ == "__main__":
    main()
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with missing values handled
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'drop':
        df = df.dropna(subset=numeric_cols)
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
    
    return df

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 11, 10, 9, 8, 12, 13, 14, 15, 16]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original data:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print(f"\nAfter outlier removal: {cleaned_df.shape}")
    
    stats = calculate_basic_stats(cleaned_df, 'values')
    print("\nBasic statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not clean column '{column}': {e}")
    
    return cleaned_df
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using Interquartile Range method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mean())
    elif strategy == 'median':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
    elif strategy == 'mode':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mode()[0])
    elif strategy == 'drop':
        data = data.dropna(subset=numeric_cols)
    
    return data

def detect_skewness(data, column, threshold=0.5):
    """
    Detect skewness in data distribution
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    skewness = stats.skew(data[column].dropna())
    is_skewed = abs(skewness) > threshold
    
    return skewness, is_skewed

def log_transform(data, column):
    """
    Apply log transformation to reduce skewness
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if (data[column] <= 0).any():
        transformed = np.log1p(data[column] - data[column].min() + 1)
    else:
        transformed = np.log(data[column])
    
    return transformed

def create_data_summary(data):
    """
    Create comprehensive data summary
    """
    summary = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(data.select_dtypes(include=['object']).columns),
        'missing_values': data.isnull().sum().to_dict(),
        'data_types': data.dtypes.to_dict()
    }
    
    return summary
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_zscore(self, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        z_scores = np.abs(stats.zscore(self.df[numeric_cols].fillna(0)))
        mask = (z_scores < threshold).all(axis=1)
        self.df = self.df[mask]
        removed_count = self.original_shape[0] - self.df.shape[0]
        print(f"Removed {removed_count} outliers using Z-score method")
        return self
        
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                col_min = self.df[col].min()
                col_max = self.df[col].max()
                if col_max > col_min:
                    self.df[col] = (self.df[col] - col_min) / (col_max - col_min)
        print(f"Normalized columns: {list(columns)}")
        return self
        
    def fill_missing_median(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
        return self
        
    def get_cleaned_data(self):
        return self.df.copy()
        
    def summary(self):
        print(f"Original shape: {self.original_shape}")
        print(f"Cleaned shape: {self.df.shape}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        print(f"Data types:\n{self.df.dtypes.value_counts()}")

def example_usage():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(1000, 50), 'feature_a'] = np.nan
    df.loc[10:15, 'feature_b'] = 1000
    
    cleaner = DataCleaner(df)
    cleaned = (cleaner
               .remove_outliers_zscore()
               .fill_missing_median()
               .normalize_minmax(['feature_a', 'feature_b'])
               .get_cleaned_data())
    
    cleaner.summary()
    return cleaned

if __name__ == "__main__":
    result = example_usage()
    print("\nFirst 5 rows of cleaned data:")
    print(result.head())import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', fill_value: Optional[float] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' to remove rows, 'fill' to replace values
        fill_value: Value to use when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        if fill_value is not None:
            return df.fillna(fill_value)
        else:
            return df.fillna(df.mean(numeric_only=True))
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase with underscores.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with cleaned column names
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Perform basic validation on DataFrame.
    
    Args:
        df: Input DataFrame
    
    Returns:
        True if DataFrame passes validation
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if df.isnull().all().any():
        raise ValueError("DataFrame contains columns with all null values")
    
    return True

def clean_data_pipeline(df: pd.DataFrame, 
                       deduplicate: bool = True,
                       handle_nulls: bool = True,
                       clean_names: bool = True) -> pd.DataFrame:
    """
    Execute a complete data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        handle_nulls: Whether to handle missing values
        clean_names: Whether to clean column names
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if clean_names:
        cleaned_df = clean_column_names(cleaned_df)
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if handle_nulls:
        cleaned_df = handle_missing_values(cleaned_df, strategy='fill')
    
    validate_dataframe(cleaned_df)
    
    return cleaned_df