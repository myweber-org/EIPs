
import pandas as pd
import numpy as np
from typing import Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[list] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset)
        return self
        
    def fill_missing_numeric(self, strategy: str = 'mean', fill_value: Optional[float] = None) -> 'DataCleaner':
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif strategy == 'constant' and fill_value is not None:
                    self.df[col].fillna(fill_value, inplace=True)
                else:
                    raise ValueError(f"Invalid strategy: {strategy}")
        
        return self
        
    def fill_missing_categorical(self, fill_value: str = 'Unknown') -> 'DataCleaner':
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if self.df[col].isnull().any():
                self.df[col].fillna(fill_value, inplace=True)
        
        return self
        
    def remove_columns_with_high_missing(self, threshold: float = 0.5) -> 'DataCleaner':
        missing_ratios = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_ratios[missing_ratios > threshold].index
        self.df = self.df.drop(columns=cols_to_drop)
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df.copy()
        
    def get_cleaning_report(self) -> dict:
        final_shape = self.df.shape
        rows_removed = self.original_shape[0] - final_shape[0]
        cols_removed = self.original_shape[1] - final_shape[1]
        
        return {
            'original_shape': self.original_shape,
            'final_shape': final_shape,
            'rows_removed': rows_removed,
            'columns_removed': cols_removed,
            'missing_values_remaining': self.df.isnull().sum().sum()
        }

def load_and_clean_csv(filepath: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(filepath, **kwargs)
    cleaner = DataCleaner(df)
    
    cleaner.remove_duplicates() \
           .remove_columns_with_high_missing(threshold=0.3) \
           .fill_missing_numeric(strategy='median') \
           .fill_missing_categorical(fill_value='Missing')
    
    report = cleaner.get_cleaning_report()
    print(f"Data cleaning completed. Removed {report['rows_removed']} rows and {report['columns_removed']} columns.")
    
    return cleaner.get_cleaned_data()