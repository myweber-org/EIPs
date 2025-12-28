
import pandas as pd
import numpy as np
from typing import List, Union

def remove_duplicates(dataframe: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        dataframe: Input DataFrame
        subset: List of column names to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return dataframe.drop_duplicates(subset=subset, keep='first')

def normalize_column(dataframe: pd.DataFrame, column: str, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize values in a specified column.
    
    Args:
        dataframe: Input DataFrame
        column: Name of column to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    df = dataframe.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            df[column] = (df[column] - mean_val) / std_val
    
    return df

def handle_missing_values(dataframe: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in numeric columns.
    
    Args:
        dataframe: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', or 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    df = dataframe.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        df = df.dropna(subset=numeric_cols)
    elif strategy == 'mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    elif strategy == 'median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    
    return df

def clean_dataset(dataframe: pd.DataFrame, 
                  deduplicate: bool = True,
                  normalize_columns: List[str] = None,
                  missing_strategy: str = 'mean') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: Input DataFrame
        deduplicate: Whether to remove duplicates
        normalize_columns: List of columns to normalize
        missing_strategy: Strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    df = dataframe.copy()
    
    if deduplicate:
        df = remove_duplicates(df)
    
    df = handle_missing_values(df, strategy=missing_strategy)
    
    if normalize_columns:
        for col in normalize_columns:
            if col in df.columns:
                df = normalize_column(df, col)
    
    return dfimport pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. 
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            If None, missing values are not filled.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = df.select_dtypes(exclude=[np.number]).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].mean()
            )
        elif strategy == 'median':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].median()
            )
        elif strategy == 'mode':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].mode().iloc[0]
            )
        elif strategy == 'constant' and fill_value is not None:
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(fill_value)
        
        self.df[self.categorical_columns] = self.df[self.categorical_columns].fillna('Unknown')
        return self

    def remove_outliers_zscore(self, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[self.numeric_columns]))
        outlier_mask = (z_scores < threshold).all(axis=1)
        self.df = self.df[outlier_mask]
        return self

    def remove_outliers_iqr(self, multiplier=1.5):
        for col in self.numeric_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self

    def standardize_data(self):
        for col in self.numeric_columns:
            mean = self.df[col].mean()
            std = self.df[col].std()
            if std > 0:
                self.df[col] = (self.df[col] - mean) / std
        return self

    def normalize_data(self):
        for col in self.numeric_columns:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            if max_val > min_val:
                self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        return self

    def get_cleaned_data(self):
        return self.df.copy()

    def get_summary(self):
        summary = {
            'original_rows': len(self.df),
            'numeric_columns': list(self.numeric_columns),
            'categorical_columns': list(self.categorical_columns),
            'missing_values': self.df.isnull().sum().sum(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def load_and_clean_data(filepath, cleaning_steps=None):
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    if cleaning_steps:
        for step in cleaning_steps:
            if step['method'] == 'handle_missing':
                cleaner.handle_missing_values(**step.get('params', {}))
            elif step['method'] == 'remove_outliers_zscore':
                cleaner.remove_outliers_zscore(**step.get('params', {}))
            elif step['method'] == 'remove_outliers_iqr':
                cleaner.remove_outliers_iqr(**step.get('params', {}))
            elif step['method'] == 'standardize':
                cleaner.standardize_data()
            elif step['method'] == 'normalize':
                cleaner.normalize_data()
    
    return cleaner.get_cleaned_data(), cleaner.get_summary()
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    df = df.dropna()
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv', ['age', 'income', 'score'])
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaning complete. Remaining records: {len(cleaned_data)}")import pandas as pd
import numpy as np
from typing import List, Union

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df: pd.DataFrame, 
                         column_types: dict) -> pd.DataFrame:
    """
    Convert columns to specified data types.
    
    Args:
        df: Input DataFrame
        column_types: Dictionary mapping column names to target types
    
    Returns:
        DataFrame with converted column types
    """
    df_copy = df.copy()
    for column, dtype in column_types.items():
        if column in df_copy.columns:
            try:
                df_copy[column] = df_copy[column].astype(dtype)
            except (ValueError, TypeError):
                df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
    return df_copy

def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = 'mean',
                         columns: List[str] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: Specific columns to process
    
    Returns:
        DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    for column in columns:
        if column not in df_copy.columns:
            continue
            
        if strategy == 'drop':
            df_copy = df_copy.dropna(subset=[column])
        elif strategy == 'mean':
            df_copy[column] = df_copy[column].fillna(df_copy[column].mean())
        elif strategy == 'median':
            df_copy[column] = df_copy[column].fillna(df_copy[column].median())
        elif strategy == 'mode':
            df_copy[column] = df_copy[column].fillna(df_copy[column].mode()[0])
    
    return df_copy

def normalize_columns(df: pd.DataFrame,
                     columns: List[str] = None,
                     method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize specified columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized columns
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns
    
    for column in columns:
        if column not in df_copy.columns:
            continue
            
        if method == 'minmax':
            col_min = df_copy[column].min()
            col_max = df_copy[column].max()
            if col_max != col_min:
                df_copy[column] = (df_copy[column] - col_min) / (col_max - col_min)
        elif method == 'zscore':
            col_mean = df_copy[column].mean()
            col_std = df_copy[column].std()
            if col_std != 0:
                df_copy[column] = (df_copy[column] - col_mean) / col_std
    
    return df_copy

def clean_dataframe(df: pd.DataFrame,
                   deduplicate: bool = True,
                   type_conversions: dict = None,
                   missing_strategy: str = 'mean',
                   normalize: bool = False,
                   normalize_method: str = 'minmax') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        type_conversions: Dictionary of column type conversions
        missing_strategy: Strategy for handling missing values
        normalize: Whether to normalize numeric columns
        normalize_method: Normalization method
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    if missing_strategy != 'none':
        cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df, method=normalize_method)
    
    return cleaned_df