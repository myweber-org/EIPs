
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self
        
    def handle_missing_values(self, strategy='mean', fill_value=None):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'mode':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mode().iloc[0])
        elif strategy == 'constant' and fill_value is not None:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_value)
            
        return self
        
    def detect_outliers_zscore(self, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        z_scores = np.abs(stats.zscore(self.df[numeric_cols]))
        outlier_mask = (z_scores > threshold).any(axis=1)
        return outlier_mask
    
    def remove_outliers(self, method='zscore', threshold=3):
        if method == 'zscore':
            outlier_mask = self.detect_outliers_zscore(threshold)
            self.df = self.df[~outlier_mask]
        return self
        
    def normalize_data(self, method='minmax'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    
        elif method == 'standard':
            for col in numeric_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
                    
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

def clean_dataset(df, remove_duplicates=True, handle_missing=True, 
                  remove_outliers=True, normalize=True):
    cleaner = DataCleaner(df)
    
    if remove_duplicates:
        cleaner.remove_duplicates()
        
    if handle_missing:
        cleaner.handle_missing_values(strategy='mean')
        
    if remove_outliers:
        cleaner.remove_outliers(threshold=3)
        
    if normalize:
        cleaner.normalize_data(method='minmax')
        
    return cleaner.get_cleaned_data(), cleaner.get_summary()import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (bool): Whether to fill missing values.
        fill_value: Value to use for filling missing data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
        multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from a DataFrame.
    """
    if subset is None:
        return df.drop_duplicates()
    else:
        return df.drop_duplicates(subset=subset)

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in specified columns using a given strategy.
    """
    if columns is None:
        columns = df.columns
    
    df_filled = df.copy()
    
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = 0
            
            df_filled[col] = df[col].fillna(fill_value)
        else:
            df_filled[col] = df[col].fillna('Unknown')
    
    return df_filled

def normalize_column(df, column, method='minmax'):
    """
    Normalize a numerical column using specified method.
    """
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df[column] = (df[column] - mean_val) / std_val
    
    return df

def clean_dataframe(df, operations=None):
    """
    Apply multiple cleaning operations to a DataFrame.
    """
    if operations is None:
        operations = []
    
    cleaned_df = df.copy()
    
    for operation in operations:
        if operation['type'] == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df, operation.get('subset'))
        elif operation['type'] == 'fill_missing':
            cleaned_df = fill_missing_values(
                cleaned_df, 
                operation.get('strategy', 'mean'),
                operation.get('columns')
            )
        elif operation['type'] == 'normalize':
            cleaned_df = normalize_column(
                cleaned_df,
                operation['column'],
                operation.get('method', 'minmax')
            )
    
    return cleaned_df

def validate_dataframe(df, rules=None):
    """
    Validate DataFrame against a set of rules.
    """
    if rules is None:
        return True, []
    
    errors = []
    
    for rule in rules:
        if rule['type'] == 'not_null':
            null_count = df[rule['column']].isnull().sum()
            if null_count > 0:
                errors.append(f"Column {rule['column']} has {null_count} null values")
        
        elif rule['type'] == 'unique':
            duplicate_count = df[rule['column']].duplicated().sum()
            if duplicate_count > 0:
                errors.append(f"Column {rule['column']} has {duplicate_count} duplicate values")
        
        elif rule['type'] == 'range':
            min_val = rule.get('min')
            max_val = rule.get('max')
            
            if min_val is not None:
                below_min = (df[rule['column']] < min_val).sum()
                if below_min > 0:
                    errors.append(f"Column {rule['column']} has {below_min} values below minimum {min_val}")
            
            if max_val is not None:
                above_max = (df[rule['column']] > max_val).sum()
                if above_max > 0:
                    errors.append(f"Column {rule['column']} has {above_max} values above maximum {max_val}")
    
    return len(errors) == 0, errors
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
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
        'count': len(df[column])
    }
    
    return stats

def example_usage():
    """
    Example demonstrating how to use the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:", calculate_summary_statistics(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_df, 'value'))
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        
        removed_count = len(self.df) - len(clean_df)
        self.df = clean_df
        return removed_count
    
    def zscore_normalize(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in normalized_df.columns:
                mean = normalized_df[col].mean()
                std = normalized_df[col].std()
                if std > 0:
                    normalized_df[col] = (normalized_df[col] - mean) / std
        
        self.df = normalized_df
        return self
    
    def minmax_normalize(self, columns=None, feature_range=(0, 1)):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        min_val, max_val = feature_range
        
        for col in columns:
            if col in normalized_df.columns:
                col_min = normalized_df[col].min()
                col_max = normalized_df[col].max()
                col_range = col_max - col_min
                
                if col_range > 0:
                    normalized_df[col] = min_val + (normalized_df[col] - col_min) * (max_val - min_val) / col_range
        
        self.df = normalized_df
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in filled_df.columns and filled_df[col].isnull().any():
                median_val = filled_df[col].median()
                filled_df[col].fillna(median_val, inplace=True)
        
        self.df = filled_df
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        return {
            'original_rows': self.original_shape[0],
            'cleaned_rows': len(self.df),
            'removed_rows': self.original_shape[0] - len(self.df),
            'columns': list(self.df.columns)
        }