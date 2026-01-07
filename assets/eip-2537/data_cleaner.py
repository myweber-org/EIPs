import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
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
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to analyze.
    
    Returns:
        dict: Dictionary containing summary statistics.
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

def process_dataset(file_path, column_name):
    """
    Load a dataset, remove outliers, and return cleaned data with statistics.
    
    Args:
        file_path (str): Path to the CSV file.
        column_name (str): Column to clean.
    
    Returns:
        tuple: (cleaned DataFrame, statistics dictionary)
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    cleaned_data = remove_outliers_iqr(data, column_name)
    stats = calculate_summary_statistics(cleaned_data, column_name)
    
    return cleaned_data, stats

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.random.normal(300, 50, 5)
        ])
    })
    
    cleaned_df = remove_outliers_iqr(sample_data, 'values')
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    
    stats = calculate_summary_statistics(cleaned_df, 'values')
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        return self
        
    def handle_missing_values(self, 
                             strategy: str = 'mean',
                             columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                elif strategy == 'zero':
                    self.df[col].fillna(0, inplace=True)
                    
        return self
        
    def remove_outliers_iqr(self, 
                           columns: Optional[List[str]] = None,
                           multiplier: float = 1.5) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & 
                                 (self.df[col] <= upper_bound)]
                
        return self
        
    def standardize_columns(self, 
                           columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in columns:
            if col in self.df.columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    self.df[col] = (self.df[col] - mean) / std
                    
        return self
        
    def normalize_columns(self, 
                         columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in columns:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    
        return self
        
    def get_cleaning_report(self) -> Dict[str, Union[int, float]]:
        current_shape = self.df.shape
        rows_removed = self.original_shape[0] - current_shape[0]
        cols_removed = self.original_shape[1] - current_shape[1]
        
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'current_rows': current_shape[0],
            'current_columns': current_shape[1],
            'rows_removed': rows_removed,
            'columns_removed': cols_removed,
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': self.df.duplicated().sum()
        }
        
    def get_dataframe(self) -> pd.DataFrame:
        return self.df.copy()
        
    def save_cleaned_data(self, filepath: str, **kwargs) -> None:
        self.df.to_csv(filepath, **kwargs)

def clean_csv_file(input_path: str, 
                  output_path: str,
                  missing_strategy: str = 'mean',
                  remove_outliers: bool = True) -> Dict[str, Union[int, float]]:
    df = pd.read_csv(input_path)
    cleaner = DataCleaner(df)
    
    cleaner.remove_duplicates()
    cleaner.handle_missing_values(strategy=missing_strategy)
    
    if remove_outliers:
        cleaner.remove_outliers_iqr()
        
    cleaner.standardize_columns()
    cleaner.save_cleaned_data(output_path, index=False)
    
    return cleaner.get_cleaning_report()