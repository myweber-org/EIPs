
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
    
    return validationimport pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        self.load_data()
    
    def load_data(self):
        if self.file_path.exists():
            self.df = pd.read_csv(self.file_path)
            print(f"Loaded data with shape: {self.df.shape}")
        else:
            raise FileNotFoundError(f"File not found: {self.file_path}")
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            raise ValueError("No data loaded")
        
        if columns is None:
            columns = self.df.columns
        
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                    fill_value = self.df[col].mean()
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                    continue
                else:
                    fill_value = 0
                
                self.df[col] = self.df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' using {strategy} strategy")
    
    def remove_duplicates(self, subset=None, keep='first'):
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
    
    def normalize_numeric(self, columns=None):
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            columns = list(numeric_cols)
        
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    print(f"Normalized column '{col}' to range [0, 1]")
    
    def save_cleaned_data(self, output_path=None):
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to: {output_path}")
        return output_path
    
    def get_summary(self):
        summary = {
            'original_file': str(self.file_path),
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': len(self.df) - len(self.df.drop_duplicates()),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def clean_csv_file(input_file, output_file=None, missing_strategy='mean'):
    cleaner = DataCleaner(input_file)
    summary_before = cleaner.get_summary()
    
    cleaner.handle_missing_values(strategy=missing_strategy)
    cleaner.remove_duplicates()
    cleaner.normalize_numeric()
    
    output_path = cleaner.save_cleaned_data(output_file)
    summary_after = cleaner.get_summary()
    
    print("\nCleaning Summary:")
    print(f"Missing values fixed: {summary_before['missing_values']} -> {summary_after['missing_values']}")
    print(f"Data shape: {summary_before['rows']}x{summary_before['columns']} -> {summary_after['rows']}x{summary_after['columns']}")
    
    return output_path

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 10.5],
        'category': ['A', 'B', None, 'A', 'C', 'A']
    }
    
    test_file = Path("test_data.csv")
    pd.DataFrame(sample_data).to_csv(test_file, index=False)
    
    try:
        result = clean_csv_file(test_file, missing_strategy='mean')
        print(f"\nCleaned file saved to: {result}")
    finally:
        if test_file.exists():
            test_file.unlink()
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize column using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column]
    
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    """
    Standardize column using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column]
    
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_threshold=1.5, normalize_method='minmax'):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        df: input DataFrame
        numeric_columns: list of numeric column names
        outlier_threshold: IQR threshold for outlier removal
        normalize_method: 'minmax' or 'zscore'
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
            
            if normalize_method == 'minmax':
                cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
            elif normalize_method == 'zscore':
                cleaned_df[f'{col}_standardized'] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, min_rows=10):
    """
    Validate dataset structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: list of required column names
        min_rows: minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(df) < min_rows:
        return False, f"Dataset has fewer than {min_rows} rows"
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return False, "No numeric columns found in dataset"
    
    return True, "Dataset validation passed"

def get_summary_statistics(df):
    """
    Generate summary statistics for numeric columns.
    
    Args:
        df: input DataFrame
    
    Returns:
        DataFrame with summary statistics
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    summary = numeric_df.describe().T
    summary['missing'] = numeric_df.isnull().sum()
    summary['zeros'] = (numeric_df == 0).sum()
    
    return summary
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    filtered_data = data.iloc[filtered_indices]
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

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        if normalize_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f'{column}_standardized'] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def get_summary_statistics(data, numeric_columns):
    """
    Generate summary statistics for numeric columns
    """
    summary = {}
    
    for column in numeric_columns:
        if column in data.columns:
            col_data = data[column].dropna()
            summary[column] = {
                'count': len(col_data),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                '25%': col_data.quantile(0.25),
                '50%': col_data.quantile(0.50),
                '75%': col_data.quantile(0.75),
                'max': col_data.max(),
                'skewness': col_data.skew(),
                'kurtosis': col_data.kurtosis()
            }
    
    return pd.DataFrame(summary).T

def handle_missing_values(data, strategy='mean', numeric_columns=None):
    """
    Handle missing values in the dataset
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    filled_data = data.copy()
    
    for column in numeric_columns:
        if column not in filled_data.columns:
            continue
            
        if strategy == 'mean':
            fill_value = filled_data[column].mean()
        elif strategy == 'median':
            fill_value = filled_data[column].median()
        elif strategy == 'mode':
            fill_value = filled_data[column].mode()[0]
        elif strategy == 'zero':
            fill_value = 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        filled_data[column] = filled_data[column].fillna(fill_value)
    
    return filled_data