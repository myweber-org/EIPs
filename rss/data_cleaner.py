import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
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

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        df[column + '_normalized'] = 0.5
    else:
        df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    
    return df

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df.dropna(subset=numeric_cols)
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df[col] = df[col].fillna(fill_value)
    
    return df

def clean_dataset(df, numeric_columns=None, outlier_removal=True, normalization=True, missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric columns to process
    outlier_removal (bool): Whether to remove outliers
    normalization (bool): Whether to normalize columns
    missing_strategy (str): Strategy for handling missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if numeric_columns is None:
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy)
    
    if outlier_removal:
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean = remove_outliers_iqr(df_clean, col)
    
    if normalization:
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean = normalize_column(df_clean, col)
    
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    }
    
    df_sample = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df_sample)
    
    cleaned_df = clean_dataset(df_sample, outlier_removal=True, normalization=True)
    print("\nCleaned DataFrame:")
    print(cleaned_df)import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset)
        return self
        
    def handle_missing_values(self, strategy: str = 'mean', 
                              columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.columns
            
        for col in columns:
            if self.df[col].dtype in ['int64', 'float64']:
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                    
                self.df[col] = self.df[col].fillna(fill_value)
            else:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                
        return self
        
    def convert_types(self, type_map: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_map.items():
            if col in self.df.columns:
                if dtype == 'datetime':
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                else:
                    self.df[col] = self.df[col].astype(dtype, errors='ignore')
        return self
        
    def remove_outliers(self, column: str, method: str = 'iqr', 
                        threshold: float = 1.5) -> 'DataCleaner':
        if column not in self.df.columns:
            return self
            
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            self.df = self.df[(self.df[column] >= lower_bound) & 
                             (self.df[column] <= upper_bound)]
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(self.df[column]))
            self.df = self.df[z_scores < threshold]
            
        return self
        
    def normalize_column(self, column: str, method: str = 'minmax') -> 'DataCleaner':
        if column not in self.df.columns:
            return self
            
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'standard':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val > 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
                
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_cleaning_report(self) -> Dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }

def clean_csv_file(input_path: str, output_path: str, 
                   cleaning_steps: Optional[Dict] = None) -> Dict:
    df = pd.read_csv(input_path)
    cleaner = DataCleaner(df)
    
    if cleaning_steps:
        if cleaning_steps.get('remove_duplicates'):
            cleaner.remove_duplicates(cleaning_steps.get('duplicate_subset'))
            
        if cleaning_steps.get('handle_missing'):
            cleaner.handle_missing_values(
                strategy=cleaning_steps.get('missing_strategy', 'mean'),
                columns=cleaning_steps.get('missing_columns')
            )
            
        if cleaning_steps.get('convert_types'):
            cleaner.convert_types(cleaning_steps['type_map'])
            
        if cleaning_steps.get('remove_outliers'):
            for col_config in cleaning_steps['outlier_columns']:
                cleaner.remove_outliers(**col_config)
                
        if cleaning_steps.get('normalize'):
            for norm_config in cleaning_steps['normalize_columns']:
                cleaner.normalize_column(**norm_config)
    
    cleaned_df = cleaner.get_cleaned_data()
    cleaned_df.to_csv(output_path, index=False)
    
    return cleaner.get_cleaning_report()