import pandas as pd
import numpy as np
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
        
        removed_count = len(self.df) - len(df_clean)
        self.df = df_clean
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns and df_filled[col].isnull().any():
                median_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_val)
        
        self.df = df_filled
        return self
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_removal_stats(self):
        return {
            'original_rows': self.original_shape[0],
            'cleaned_rows': len(self.df),
            'rows_removed': self.original_shape[0] - len(self.df),
            'removal_percentage': ((self.original_shape[0] - len(self.df)) / self.original_shape[0]) * 100
        }

def clean_dataset(df, outlier_columns=None, normalize=True, fill_missing=True):
    cleaner = DataCleaner(df)
    
    if fill_missing:
        cleaner.fill_missing_median()
    
    if outlier_columns:
        cleaner.remove_outliers_iqr(columns=outlier_columns)
    
    if normalize:
        cleaner.normalize_minmax()
    
    return cleaner.get_cleaned_data(), cleaner.get_removal_stats()
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fill_missing: Strategy for filling missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                else:  # mode
                    fill_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary': {}
    }
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append('DataFrame is empty')
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing required columns: {missing_cols}')
    
    validation_results['summary']['total_rows'] = len(df)
    validation_results['summary']['total_columns'] = len(df.columns)
    validation_results['summary']['missing_values'] = df.isnull().sum().sum()
    validation_results['summary']['duplicate_rows'] = df.duplicated().sum()
    
    return validation_results

def normalize_numeric_columns(df, columns=None, method='minmax'):
    """
    Normalize numeric columns in a DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: List of columns to normalize (if None, normalize all numeric columns)
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized columns
    """
    normalized_df = df.copy()
    
    if columns is None:
        columns = normalized_df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in normalized_df.columns and pd.api.types.is_numeric_dtype(normalized_df[col]):
            if method == 'minmax':
                col_min = normalized_df[col].min()
                col_max = normalized_df[col].max()
                if col_max != col_min:
                    normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            elif method == 'zscore':
                col_mean = normalized_df[col].mean()
                col_std = normalized_df[col].std()
                if col_std != 0:
                    normalized_df[col] = (normalized_df[col] - col_mean) / col_std
    
    return normalized_df