
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
        'count': df[column].count(),
        'q1': df[column].quantile(0.25),
        'q3': df[column].quantile(0.75)
    }
    
    return stats

def process_dataframe(df, numeric_columns):
    """
    Process DataFrame by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics for each column
    """
    cleaned_df = df.copy()
    all_stats = {}
    
    for col in numeric_columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, col)
            stats['outliers_removed'] = removed_count
            all_stats[col] = stats
    
    return cleaned_df, all_stats

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    # Introduce some outliers
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[95:99, 'value'] = [500, 600, -200, 700, 800]
    
    print("Original DataFrame shape:", sample_df.shape)
    print("Original statistics:")
    print(sample_df['value'].describe())
    
    cleaned_df, stats = process_dataframe(sample_df, ['value'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:")
    print(cleaned_df['value'].describe())
    print("\nProcessing stats:", stats)
import numpy as np
import pandas as pd
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
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                outliers = z_scores > threshold
                df_clean = df_clean[~outliers]
        
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
                if max_val != min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                mean_val = df_normalized[col].mean()
                std_val = df_normalized[col].std()
                if std_val > 0:
                    df_normalized[col] = (df_normalized[col] - mean_val) / std_val
        
        self.df = df_normalized
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns and df_filled[col].isnull().any():
                mean_val = df_filled[col].mean()
                df_filled[col] = df_filled[col].fillna(mean_val)
        
        self.df = df_filled
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
        current_rows = len(self.df)
        removed_rows = self.original_shape[0] - current_rows
        removal_percentage = (removed_rows / self.original_shape[0]) * 100 if self.original_shape[0] > 0 else 0
        return {
            'original_rows': self.original_shape[0],
            'current_rows': current_rows,
            'removed_rows': removed_rows,
            'removal_percentage': removal_percentage
        }
    
    def save_cleaned_data(self, filepath, format='csv'):
        if format == 'csv':
            self.df.to_csv(filepath, index=False)
        elif format == 'excel':
            self.df.to_excel(filepath, index=False)
        elif format == 'parquet':
            self.df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, factor=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def z_score_normalize(dataframe, columns=None):
    """
    Apply z-score normalization to specified columns.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col in result_df.columns and np.issubdtype(result_df[col].dtype, np.number):
            mean_val = result_df[col].mean()
            std_val = result_df[col].std()
            
            if std_val > 0:
                result_df[col] = (result_df[col] - mean_val) / std_val
            else:
                result_df[col] = 0
    
    return result_df

def min_max_normalize(dataframe, columns=None, feature_range=(0, 1)):
    """
    Apply min-max normalization to specified columns.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize
        feature_range: tuple of (min, max) for output range
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    result_df = dataframe.copy()
    min_range, max_range = feature_range
    
    for col in columns:
        if col in result_df.columns and np.issubdtype(result_df[col].dtype, np.number):
            min_val = result_df[col].min()
            max_val = result_df[col].max()
            
            if max_val > min_val:
                result_df[col] = ((result_df[col] - min_val) / (max_val - min_val)) * \
                                (max_range - min_range) + min_range
            else:
                result_df[col] = min_range
    
    return result_df

def detect_skewed_columns(dataframe, threshold=0.5):
    """
    Detect columns with skewed distributions.
    
    Args:
        dataframe: pandas DataFrame
        threshold: absolute skewness threshold (default 0.5)
    
    Returns:
        Dictionary with skewness values for columns exceeding threshold
    """
    skewed_cols = {}
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        skewness = stats.skew(dataframe[col].dropna())
        if abs(skewness) > threshold:
            skewed_cols[col] = skewness
    
    return skewed_cols

def log_transform_skewed(dataframe, skewed_columns):
    """
    Apply log transformation to reduce skewness.
    
    Args:
        dataframe: pandas DataFrame
        skewed_columns: list of column names to transform
    
    Returns:
        DataFrame with transformed columns
    """
    result_df = dataframe.copy()
    
    for col in skewed_columns:
        if col in result_df.columns:
            # Add small constant to handle zero or negative values
            min_val = result_df[col].min()
            if min_val <= 0:
                constant = abs(min_val) + 1
                result_df[col] = np.log(result_df[col] + constant)
            else:
                result_df[col] = np.log(result_df[col])
    
    return result_df

def clean_dataset(dataframe, outlier_columns=None, normalize_method='zscore', 
                  handle_skewness=True, skew_threshold=0.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame to clean
        outlier_columns: columns for outlier removal (default: all numeric)
        normalize_method: 'zscore', 'minmax', or None
        handle_skewness: whether to transform skewed columns
        skew_threshold: threshold for detecting skewness
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = dataframe.copy()
    
    # Remove outliers
    if outlier_columns is not None:
        for col in outlier_columns:
            if col in df_clean.columns:
                df_clean = remove_outliers_iqr(df_clean, col)
    
    # Handle skewness
    if handle_skewness:
        skewed = detect_skewed_columns(df_clean, skew_threshold)
        if skewed:
            df_clean = log_transform_skewed(df_clean, list(skewed.keys()))
    
    # Normalize
    if normalize_method == 'zscore':
        df_clean = z_score_normalize(df_clean)
    elif normalize_method == 'minmax':
        df_clean = min_max_normalize(df_clean)
    
    return df_clean

def validate_cleaning(dataframe, original_dataframe):
    """
    Validate cleaning process by comparing statistics.
    
    Args:
        dataframe: cleaned DataFrame
        original_dataframe: original DataFrame
    
    Returns:
        Dictionary with validation metrics
    """
    validation = {}
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in original_dataframe.columns:
            orig_mean = original_dataframe[col].mean()
            orig_std = original_dataframe[col].std()
            clean_mean = dataframe[col].mean()
            clean_std = dataframe[col].std()
            
            validation[col] = {
                'original_mean': orig_mean,
                'original_std': orig_std,
                'cleaned_mean': clean_mean,
                'cleaned_std': clean_std,
                'mean_change_percent': abs((clean_mean - orig_mean) / orig_mean * 100) if orig_mean != 0 else 0,
                'std_change_percent': abs((clean_std - orig_std) / orig_std * 100) if orig_std != 0 else 0
            }
    
    return validation