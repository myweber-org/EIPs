
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.numeric_columns
            
        clean_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)
                clean_df = clean_df[mask]
                
        return clean_df.reset_index(drop=True)
    
    def zscore_normalize(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        normalized_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val > 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
                    
        return normalized_df
    
    def minmax_normalize(self, columns=None, feature_range=(0, 1)):
        if columns is None:
            columns = self.numeric_columns
            
        normalized_df = self.df.copy()
        min_val, max_val = feature_range
        
        for col in columns:
            if col in self.numeric_columns:
                col_min = normalized_df[col].min()
                col_max = normalized_df[col].max()
                col_range = col_max - col_min
                
                if col_range > 0:
                    normalized_df[col] = (normalized_df[col] - col_min) / col_range
                    normalized_df[col] = normalized_df[col] * (max_val - min_val) + min_val
                    
        return normalized_df
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        filled_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns and filled_df[col].isnull().any():
                median_val = filled_df[col].median()
                filled_df[col] = filled_df[col].fillna(median_val)
                
        return filled_df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': list(self.numeric_columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary
import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(
    data: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first',
    reset_index: bool = True
) -> pd.DataFrame:
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame containing potential duplicates
    subset : list, optional
        Column names to consider for identifying duplicates
    keep : str, default 'first'
        Which duplicates to keep: 'first', 'last', or False
    reset_index : bool, default True
        Whether to reset the DataFrame index after removal
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with duplicates removed
    """
    if data.empty:
        return data
    
    cleaned_data = data.drop_duplicates(
        subset=subset,
        keep=keep,
        ignore_index=reset_index
    )
    
    if not reset_index:
        cleaned_data.index = data.index[~data.duplicated(subset=subset, keep=keep)]
    
    return cleaned_data

def find_duplicate_indices(
    data: pd.DataFrame,
    subset: Optional[List[str]] = None
) -> pd.Index:
    """
    Find indices of duplicate rows.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame
    subset : list, optional
        Column names to consider
    
    Returns:
    --------
    pd.Index
        Indices of duplicate rows
    """
    return data.index[data.duplicated(subset=subset, keep=False)]

def clean_numeric_outliers(
    data: pd.DataFrame,
    column: str,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers from a numeric column using specified method.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame
    column : str
        Column name to clean
    method : str, default 'iqr'
        Method for outlier detection: 'iqr' or 'zscore'
    threshold : float, default 1.5
        Threshold for outlier detection
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'iqr':
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
    
    elif method == 'zscore':
        mean = data[column].mean()
        std = data[column].std()
        z_scores = np.abs((data[column] - mean) / std)
        mask = z_scores <= threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return data[mask].copy()

def validate_data_types(
    data: pd.DataFrame,
    expected_types: dict
) -> dict:
    """
    Validate DataFrame column data types against expected types.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame
    expected_types : dict
        Dictionary mapping column names to expected dtypes
    
    Returns:
    --------
    dict
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'mismatches': [],
        'missing_columns': []
    }
    
    for col, expected_type in expected_types.items():
        if col not in data.columns:
            results['missing_columns'].append(col)
            results['valid'] = False
        elif not pd.api.types.is_dtype_equal(data[col].dtype, expected_type):
            results['mismatches'].append({
                'column': col,
                'expected': expected_type,
                'actual': str(data[col].dtype)
            })
            results['valid'] = False
    
    return results

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 1, 4, 2],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'David', 'Bob'],
        'value': [10.5, 20.3, 15.7, 10.5, 30.1, 20.3]
    })
    
    cleaned = remove_duplicates(sample_data, subset=['id', 'name'])
    print("Cleaned data:")
    print(cleaned)