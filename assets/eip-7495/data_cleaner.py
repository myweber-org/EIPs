
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df[column]),
        'q1': df[column].quantile(0.25),
        'q3': df[column].quantile(0.75)
    }
    
    return stats

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of summary statistics for each cleaned column
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_clean = list(numeric_cols)
    
    cleaned_df = df.copy()
    statistics = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            statistics[column] = stats
    
    return cleaned_df, statistics

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal summary statistics:")
    print(df.describe())
    
    cleaned_df, stats = clean_dataset(df, ['A', 'B'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaning statistics:")
    for col, col_stats in stats.items():
        print(f"\n{col}:")
        for stat_name, stat_value in col_stats.items():
            print(f"  {stat_name}: {stat_value:.2f}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
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
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                else:
                    fill_value = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value:.2f}")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].isnull().any():
                mode_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None
                if mode_value is not None:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                    print(f"Filled missing values in column '{col}' with mode: {mode_value}")
    
    print(f"Dataset cleaned: {len(df)} -> {len(cleaned_df)} rows")
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
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
        validation_results['all_required_columns_present'] = len(missing_cols) == 0
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nValidation results:")
    print(validate_dataset(df))
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned)
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
            
        clean_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                mask = (clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)
                clean_df = clean_df[mask]
                
        return clean_df.reset_index(drop=True)
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
            
        clean_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
                mask = z_scores < threshold
                clean_df = clean_df[mask]
                
        return clean_df.reset_index(drop=True)
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        normalized_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                    
        return normalized_df
    
    def normalize_zscore(self, columns=None):
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
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        filled_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                mean_val = filled_df[col].mean()
                filled_df[col] = filled_df[col].fillna(mean_val)
                
        return filled_df
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        filled_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                median_val = filled_df[col].median()
                filled_df[col] = filled_df[col].fillna(median_val)
                
        return filled_df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': self.numeric_columns,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summaryimport pandas as pd
import numpy as np
from typing import Union, List

def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers from specified columns using the IQR method.
    
    Parameters:
    df: Input DataFrame
    columns: Column name or list of column names
    multiplier: IQR multiplier (default 1.5)
    
    Returns:
    DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if isinstance(columns, str):
        columns = [columns]
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def fill_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mean',
    columns: Union[str, List[str], None] = None
) -> pd.DataFrame:
    """
    Fill missing values in DataFrame columns.
    
    Parameters:
    df: Input DataFrame
    strategy: 'mean', 'median', 'mode', or 'constant'
    columns: Specific columns to fill (None for all numeric columns)
    
    Returns:
    DataFrame with filled missing values
    """
    df_filled = df.copy()
    
    if columns is None:
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    elif isinstance(columns, str):
        columns = [columns]
    
    for col in columns:
        if col not in df_filled.columns:
            continue
            
        if strategy == 'mean':
            fill_value = df_filled[col].mean()
        elif strategy == 'median':
            fill_value = df_filled[col].median()
        elif strategy == 'mode':
            fill_value = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 0
        elif strategy == 'constant':
            fill_value = 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        df_filled[col] = df_filled[col].fillna(fill_value)
    
    return df_filled

def normalize_columns(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    method: str = 'minmax'
) -> pd.DataFrame:
    """
    Normalize specified columns in DataFrame.
    
    Parameters:
    df: Input DataFrame
    columns: Column name or list of column names
    method: 'minmax' or 'zscore'
    
    Returns:
    DataFrame with normalized columns
    """
    df_norm = df.copy()
    
    if isinstance(columns, str):
        columns = [columns]
    
    for col in columns:
        if col not in df_norm.columns:
            continue
            
        if method == 'minmax':
            col_min = df_norm[col].min()
            col_max = df_norm[col].max()
            if col_max != col_min:
                df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
        
        elif method == 'zscore':
            col_mean = df_norm[col].mean()
            col_std = df_norm[col].std()
            if col_std != 0:
                df_norm[col] = (df_norm[col] - col_mean) / col_std
    
    return df_norm

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    min_rows: int = 1
) -> bool:
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df: DataFrame to validate
    required_columns: List of required column names
    min_rows: Minimum number of rows required
    
    Returns:
    True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if len(df) < min_rows:
        return False
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False
    
    return True
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
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
    
    return filtered_df.reset_index(drop=True)

def clean_numeric_data(df, columns=None):
    """
    Clean numeric columns by removing outliers from specified columns or all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list, optional): List of column names to clean. If None, cleans all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not clean column '{column}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 21),
        'value': [10, 12, 11, 15, 9, 100, 13, 14, 12, 11, 
                  10, 9, 8, 200, 12, 13, 14, 15, 16, 17]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    cleaned_df = clean_numeric_data(df, columns=['value'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print(f"\nCleaned shape: {cleaned_df.shape}")
    
    outliers_removed = df.shape[0] - cleaned_df.shape[0]
    print(f"\nOutliers removed: {outliers_removed}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        if normalize_method == 'minmax':
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[column] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan and data.isnull().any().any():
        raise ValueError("Dataset contains NaN values")
    
    return True

def get_data_summary(data):
    """
    Generate statistical summary of the dataset
    """
    summary = {
        'shape': data.shape,
        'columns': data.columns.tolist(),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max(),
            'median': data[col].median()
        }
    
    return summary
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Remove outliers using z-score
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(exclude=[np.number]).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mean())
        elif strategy == 'median' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].median())
        elif strategy == 'mode':
            for col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else None)
        elif strategy == 'constant' and fill_value is not None:
            self.df = self.df.fillna(fill_value)
        elif strategy == 'drop':
            self.df = self.df.dropna()
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

    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        for col in columns:
            if col in self.numeric_columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    self.df[col] = (self.df[col] - mean) / std
        return self

    def normalize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        for col in columns:
            if col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        return self

    def get_cleaned_data(self):
        return self.df

def clean_dataset(df, missing_strategy='mean', remove_outliers=True, standardize=False, normalize=False):
    cleaner = DataCleaner(df)
    cleaner.handle_missing_values(strategy=missing_strategy)
    if remove_outliers:
        cleaner.remove_outliers_iqr()
    if standardize:
        cleaner.standardize_data()
    if normalize:
        cleaner.normalize_data()
    return cleaner.get_cleaned_data()