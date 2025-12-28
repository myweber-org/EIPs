
import pandas as pd
import numpy as np
from typing import Optional, Union, List

def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    factor: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers from specified columns using IQR method.
    
    Parameters:
    df: Input DataFrame
    columns: Column name or list of column names to process
    factor: IQR multiplier for outlier detection
    
    Returns:
    DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    elif isinstance(columns, str):
        columns = [columns]
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mean',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Parameters:
    df: Input DataFrame
    strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns: Specific columns to process
    
    Returns:
    DataFrame with handled missing values
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'drop':
            df_filled = df_filled.dropna(subset=[col])
        elif strategy in ['mean', 'median']:
            if pd.api.types.is_numeric_dtype(df[col]):
                if strategy == 'mean':
                    fill_value = df[col].mean()
                else:
                    fill_value = df[col].median()
                df_filled[col] = df_filled[col].fillna(fill_value)
        elif strategy == 'mode':
            if not df[col].empty:
                fill_value = df[col].mode()[0] if not df[col].mode().empty else None
                df_filled[col] = df_filled[col].fillna(fill_value)
    
    return df_filled.reset_index(drop=True)

def normalize_data(
    df: pd.DataFrame,
    method: str = 'minmax',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Normalize numerical columns in DataFrame.
    
    Parameters:
    df: Input DataFrame
    method: Normalization method ('minmax', 'zscore')
    columns: Specific columns to normalize
    
    Returns:
    DataFrame with normalized columns
    """
    df_norm = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df_norm[col] = (df[col] - mean_val) / std_val
    
    return df_norm

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1
) -> bool:
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df: DataFrame to validate
    required_columns: List of columns that must be present
    min_rows: Minimum number of rows required
    
    Returns:
    Boolean indicating if DataFrame is valid
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if len(df) < min_rows:
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False
    
    return Trueimport pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop')
        columns (list): List of columns to process, None for all numeric columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=columns)
    elif strategy == 'mean':
        for col in columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    elif strategy == 'median':
        for col in columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif strategy == 'mode':
        for col in columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    return df_clean

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to process, None for all numeric columns
        threshold (float): IQR multiplier threshold
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    df_standardized = df.copy()
    
    if columns is None:
        columns = df_standardized.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        mean = df_standardized[col].mean()
        std = df_standardized[col].std()
        if std > 0:
            df_standardized[col] = (df_standardized[col] - mean) / std
    
    return df_standardized

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): If True, remove duplicate rows.
    fill_method (str): Method to handle missing values: 'drop' to remove rows,
                       'fill_mean' to fill numeric columns with mean,
                       'fill_median' to fill numeric columns with median,
                       'fill_mode' to fill with mode for categorical columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_method == 'fill_mean':
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    elif fill_method == 'fill_median':
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_method == 'fill_mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                mode_val = cleaned_df[col].mode()
                if not mode_val.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_cols
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, None, 4, 5, 5],
        'B': [10, 20, 30, None, 50, 50],
        'C': ['x', 'y', 'z', 'x', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation results:")
    print(validate_dataset(df))
    
    cleaned = clean_dataset(df, remove_duplicates=True, fill_method='fill_mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nValidation after cleaning:")
    print(validate_dataset(cleaned))
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std()
    }
    return statsimport pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = df.select_dtypes(exclude=[np.number]).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].mean()
            )
        elif strategy == 'median' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].median()
            )
        elif strategy == 'mode':
            for col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        elif strategy == 'constant' and fill_value is not None:
            self.df = self.df.fillna(fill_value)
        return self

    def remove_outliers(self, method='zscore', threshold=3):
        if method == 'zscore' and self.numeric_columns.any():
            z_scores = np.abs(stats.zscore(self.df[self.numeric_columns]))
            mask = (z_scores < threshold).all(axis=1)
            self.df = self.df[mask]
        elif method == 'iqr' and self.numeric_columns.any():
            for col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self

    def normalize_data(self, method='minmax'):
        if method == 'minmax' and self.numeric_columns.any():
            for col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        elif method == 'standard' and self.numeric_columns.any():
            for col in self.numeric_columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        return self

    def get_cleaned_data(self):
        return self.df

def create_sample_data():
    data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['X', 'Y', 'X', 'Y', np.nan, 'X']
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_sample_data()
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                  .handle_missing_values(strategy='mean')
                  .remove_outliers(method='zscore', threshold=2.5)
                  .normalize_data(method='minmax')
                  .get_cleaned_data())
    print("Original data shape:", df.shape)
    print("Cleaned data shape:", cleaned_df.shape)
    print("\nCleaned data:")
    print(cleaned_df)
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = df.select_dtypes(include=['object']).columns
    
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
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        return self
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                valid_indices = np.where(z_scores < threshold)[0]
                original_indices = self.df[col].dropna().index[valid_indices]
                self.df = self.df.loc[original_indices.union(self.df[self.df[col].isna()].index)]
        
        return self
    
    def normalize_data(self, method='minmax'):
        if method == 'minmax':
            for col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            for col in self.numeric_columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.numeric_columns),
            'categorical_columns': list(self.categorical_columns)
        }
        return summary

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', normalize=False):
    cleaner = DataCleaner(df)
    cleaner.handle_missing_values(strategy=missing_strategy)
    
    if outlier_method == 'iqr':
        cleaner.remove_outliers_iqr()
    elif outlier_method == 'zscore':
        cleaner.remove_outliers_zscore()
    
    if normalize:
        cleaner.normalize_data()
    
    return cleaner.get_cleaned_data()