
import pandas as pd
import numpy as np
from typing import Optional, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
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
        
    def fill_missing_categorical(self, fill_value: str = 'Unknown') -> 'DataCleaner':
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        self.df[categorical_cols] = self.df[categorical_cols].fillna(fill_value)
        return self
        
    def remove_columns_with_high_missing(self, threshold: float = 0.5) -> 'DataCleaner':
        missing_ratio = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index
        self.df = self.df.drop(columns=cols_to_drop)
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_cleaning_report(self) -> dict:
        final_shape = self.df.shape
        rows_removed = self.original_shape[0] - final_shape[0]
        cols_removed = self.original_shape[1] - final_shape[1]
        missing_values = self.df.isnull().sum().sum()
        
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'final_rows': final_shape[0],
            'final_columns': final_shape[1],
            'rows_removed': rows_removed,
            'columns_removed': cols_removed,
            'remaining_missing_values': missing_values
        }

def clean_csv_file(input_path: str, output_path: str, **kwargs) -> dict:
    df = pd.read_csv(input_path)
    cleaner = DataCleaner(df)
    
    if 'remove_duplicates' in kwargs and kwargs['remove_duplicates']:
        cleaner.remove_duplicates(kwargs.get('duplicate_subset'))
        
    if 'fill_numeric' in kwargs and kwargs['fill_numeric']:
        cleaner.fill_missing_numeric(
            strategy=kwargs.get('numeric_strategy', 'mean'),
            fill_value=kwargs.get('numeric_fill_value')
        )
        
    if 'fill_categorical' in kwargs and kwargs['fill_categorical']:
        cleaner.fill_missing_categorical(
            fill_value=kwargs.get('categorical_fill_value', 'Unknown')
        )
        
    if 'remove_high_missing' in kwargs and kwargs['remove_high_missing']:
        cleaner.remove_columns_with_high_missing(
            threshold=kwargs.get('missing_threshold', 0.5)
        )
    
    cleaned_df = cleaner.get_cleaned_data()
    cleaned_df.to_csv(output_path, index=False)
    
    return cleaner.get_cleaning_report()