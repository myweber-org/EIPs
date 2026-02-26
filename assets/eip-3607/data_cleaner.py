import numpy as np
import pandas as pd
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

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
            cleaned_df = standardize_zscore(cleaned_df, col)
    return cleaned_df

def main():
    sample_data = {
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    }
    df = pd.DataFrame(sample_data)
    print("Original dataset shape:", df.shape)
    
    numeric_cols = ['feature_a', 'feature_b']
    cleaned_df = clean_dataset(df, numeric_cols)
    print("Cleaned dataset shape:", cleaned_df.shape)
    print("Cleaned columns:", cleaned_df.columns.tolist())

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, and max.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'count': df[column].count(),
        'mean': df[column].mean(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max()
    }
    
    return stats

def clean_dataset(df, numeric_columns):
    """
    Clean dataset by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_columns (list): List of column names to clean.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame with outliers removed.
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal statistics for column 'A':")
    print(calculate_basic_stats(df, 'A'))
    
    cleaned_df = clean_dataset(df, ['A', 'B', 'C'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned statistics for column 'A':")
    print(calculate_basic_stats(cleaned_df, 'A'))
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list: List containing potentially duplicate items.
    
    Returns:
        List with duplicates removed.
    """
    seen = set()
    result = []
    
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_numeric_data(values, default=0):
    """
    Clean numeric data by converting strings to floats and handling invalid values.
    
    Args:
        values: List of values to clean.
        default: Default value to use for invalid entries.
    
    Returns:
        List of cleaned numeric values.
    """
    cleaned = []
    
    for value in values:
        try:
            if isinstance(value, str):
                cleaned.append(float(value.strip()))
            else:
                cleaned.append(float(value))
        except (ValueError, TypeError):
            cleaned.append(default)
    
    return cleaned

def filter_by_threshold(data, threshold, key=None):
    """
    Filter data based on a threshold value.
    
    Args:
        data: List of items to filter.
        threshold: Minimum value to include.
        key: Function to extract value from items if data contains dictionaries.
    
    Returns:
        Filtered list containing only items meeting the threshold.
    """
    if key is None:
        return [item for item in data if item >= threshold]
    
    return [item for item in data if key(item) >= threshold]

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
    
    numeric_data = ["10.5", "invalid", "20.3", 15, None]
    cleaned_numeric = clean_numeric_data(numeric_data)
    print(f"Numeric cleaned: {cleaned_numeric}")
    
    scores = [{"name": "A", "score": 85}, {"name": "B", "score": 92}, {"name": "C", "score": 78}]
    high_scores = filter_by_threshold(scores, 80, key=lambda x: x["score"])
    print(f"High scores: {high_scores}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    """
    Normalize column values to range [0, 1] using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized values
    """
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        return df[column].apply(lambda x: 0.5)
    
    return (df[column] - min_val) / (max_val - min_val)

def standardize_zscore(df, column):
    """
    Standardize column values using z-score normalization.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    column (str): Column name to standardize
    
    Returns:
    pd.Series: Standardized values
    """
    mean_val = df[column].mean()
    std_val = df[column].std()
    
    if std_val == 0:
        return df[column].apply(lambda x: 0)
    
    return (df[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_multiplier=1.5, normalization_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    outlier_multiplier (float): IQR multiplier for outlier detection
    normalization_method (str): 'minmax' or 'zscore' normalization
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column not in cleaned_df.columns:
            continue
            
        cleaned_df = remove_outliers_iqr(cleaned_df, column, outlier_multiplier)
        
        if normalization_method == 'minmax':
            cleaned_df[f'{column}_normalized'] = normalize_minmax(cleaned_df, column)
        elif normalization_method == 'zscore':
            cleaned_df[f'{column}_standardized'] = standardize_zscore(cleaned_df, column)
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df, required_columns):
    """
    Validate dataframe structure and required columns.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True

def get_summary_statistics(df, numeric_columns):
    """
    Calculate summary statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    summary = pd.DataFrame()
    
    for column in numeric_columns:
        if column in df.columns:
            col_stats = {
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'count': df[column].count(),
                'missing': df[column].isnull().sum()
            }
            summary[column] = pd.Series(col_stats)
    
    return summary.T
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from a column using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: multiplier for IQR (default 1.5)
    
    Returns:
        DataFrame with outliers removed
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

def normalize_minmax(data, column):
    """
    Normalize column values to range [0, 1] using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        data[column + '_normalized'] = 0.5
    else:
        data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    
    return data

def standardize_zscore(data, column):
    """
    Standardize column values using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        data[column + '_standardized'] = 0
    else:
        data[column + '_standardized'] = (data[column] - mean_val) / std_val
    
    return data

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names to process
        outlier_factor: factor for IQR outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_factor)
            cleaned_data = normalize_minmax(cleaned_data, column)
            cleaned_data = standardize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
        allow_nan: whether to allow NaN values
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if not allow_nan and data.isnull().any().any():
        nan_columns = data.columns[data.isnull().any()].tolist()
        return False, f"NaN values found in columns: {nan_columns}"
    
    return True, "Data validation passed"
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
        
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        self.df = self.df[z_scores < threshold]
        return self
        
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val > min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'standard':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val > 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
        return self
        
    def fill_missing(self, column, strategy='mean'):
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = strategy
            
        self.df[column].fillna(fill_value, inplace=True)
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]