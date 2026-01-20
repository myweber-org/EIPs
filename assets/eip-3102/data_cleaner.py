
import numpy as np
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
            col_min = self.df[column].min()
            col_max = self.df[column].max()
            if col_max != col_min:
                self.df[f'{column}_normalized'] = (self.df[column] - col_min) / (col_max - col_min)
            else:
                self.df[f'{column}_normalized'] = 0.5
        
        elif method == 'zscore':
            col_mean = self.df[column].mean()
            col_std = self.df[column].std()
            if col_std > 0:
                self.df[f'{column}_normalized'] = (self.df[column] - col_mean) / col_std
            else:
                self.df[f'{column}_normalized'] = 0
        
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'")
        
        return self.df[f'{column}_normalized']
    
    def fill_missing_values(self, strategy='mean', custom_value=None):
        df_filled = self.df.copy()
        
        for col in df_filled.columns:
            if df_filled[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_filled[col]):
                    if strategy == 'mean':
                        fill_value = df_filled[col].mean()
                    elif strategy == 'median':
                        fill_value = df_filled[col].median()
                    elif strategy == 'mode':
                        fill_value = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 0
                    elif strategy == 'custom' and custom_value is not None:
                        fill_value = custom_value
                    else:
                        continue
                    
                    df_filled[col] = df_filled[col].fillna(fill_value)
                else:
                    if strategy == 'mode':
                        fill_value = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 'Unknown'
                    elif strategy == 'custom' and custom_value is not None:
                        fill_value = custom_value
                    else:
                        fill_value = 'Unknown'
                    
                    df_filled[col] = df_filled[col].fillna(fill_value)
        
        self.df = df_filled
        return self.df
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object', 'category']).columns)
        }
        return summary

def clean_dataset(df, outlier_threshold=1.5, normalize_cols=None, fill_missing=True):
    cleaner = DataCleaner(df)
    
    if fill_missing:
        cleaner.fill_missing_values(strategy='mean')
    
    removed = cleaner.remove_outliers_iqr(threshold=outlier_threshold)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaner.df.columns:
                cleaner.normalize_column(col, method='zscore')
    
    summary = cleaner.get_summary()
    cleaned_df = cleaner.get_cleaned_data()
    
    print(f"Removed {removed} outliers")
    print(f"Final dataset shape: {cleaned_df.shape}")
    
    return cleaned_df, summary