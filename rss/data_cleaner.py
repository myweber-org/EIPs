
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column in a dataset using the IQR method.
    
    Parameters:
    data (np.ndarray or list): The dataset.
    column (int): Index of the column to process.
    
    Returns:
    np.ndarray: Data with outliers removed from the specified column.
    """
    data = np.array(data)
    col_data = data[:, column]
    
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    
    return data[mask]
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column in a dataset using the IQR method.
    
    Parameters:
    data (np.ndarray): The dataset.
    column (int): Index of the column to process.
    
    Returns:
    np.ndarray: Dataset with outliers removed from the specified column.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1] or column < 0:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    return data[mask]
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        return self
        
    def handle_missing_values(self, strategy: str = 'drop', fill_value: Optional[float] = None) -> 'DataCleaner':
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            if fill_value is not None:
                self.df = self.df.fillna(fill_value)
            else:
                self.df = self.df.fillna(self.df.mean(numeric_only=True))
        return self
        
    def normalize_column(self, column: str, method: str = 'minmax') -> 'DataCleaner':
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        if method == 'minmax':
            col_min = self.df[column].min()
            col_max = self.df[column].max()
            if col_max != col_min:
                self.df[column] = (self.df[column] - col_min) / (col_max - col_min)
        elif method == 'zscore':
            col_mean = self.df[column].mean()
            col_std = self.df[column].std()
            if col_std > 0:
                self.df[column] = (self.df[column] - col_mean) / col_std
        return self
        
    def remove_outliers(self, column: str, threshold: float = 3.0) -> 'DataCleaner':
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
        self.df = self.df[z_scores < threshold]
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_cleaning_report(self) -> dict:
        cleaned_shape = self.df.shape
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': cleaned_shape[0],
            'cleaned_columns': cleaned_shape[1],
            'rows_removed': self.original_shape[0] - cleaned_shape[0],
            'columns_removed': self.original_shape[1] - cleaned_shape[1]
        }import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using the Interquartile Range method.
    Returns filtered DataFrame and outlier indices.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
    outliers = data[~mask].index.tolist()
    
    return data[mask].copy(), outliers

def normalize_minmax(data, columns=None):
    """
    Normalize specified columns using Min-Max scaling.
    If columns is None, normalize all numeric columns.
    """
    if columns is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    normalized_data = data.copy()
    for col in columns:
        if col not in normalized_data.columns:
            continue
        col_min = normalized_data[col].min()
        col_max = normalized_data[col].max()
        if col_max > col_min:
            normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
    
    return normalized_data

def detect_anomalies_zscore(data, column, threshold=3):
    """
    Detect anomalies using Z-score method.
    Returns indices of anomalous values.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    anomaly_indices = data[column].dropna().index[z_scores > threshold].tolist()
    
    return anomaly_indices

def clean_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    cleaned_data = data.copy()
    
    if columns is None:
        columns = cleaned_data.columns
    
    for col in columns:
        if col not in cleaned_data.columns:
            continue
        
        if strategy == 'drop':
            cleaned_data = cleaned_data.dropna(subset=[col])
        elif strategy == 'mean':
            cleaned_data[col].fillna(cleaned_data[col].mean(), inplace=True)
        elif strategy == 'median':
            cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
        elif strategy == 'mode':
            cleaned_data[col].fillna(cleaned_data[col].mode()[0], inplace=True)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
    
    return cleaned_data

def validate_dataframe(data, required_columns=None, numeric_columns=None):
    """
    Basic DataFrame validation.
    Checks for required columns and numeric column types.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if required_columns:
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    if numeric_columns:
        non_numeric = [col for col in numeric_columns 
                      if not pd.api.types.is_numeric_dtype(data[col])]
        if non_numeric:
            raise ValueError(f"Non-numeric columns specified as numeric: {non_numeric}")
    
    return True