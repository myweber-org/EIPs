
import pandas as pd
import re

def clean_dataframe(df, column_names=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    """
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - df_clean.shape[0]
    
    # If specific columns are provided, clean only those
    # Otherwise, clean all object/string columns
    if column_names is None:
        column_names = df_clean.select_dtypes(include=['object']).columns
    
    for col in column_names:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(_normalize_string)
    
    return df_clean, removed_duplicates

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    """
    if pd.isna(text):
        return text
    
    # Convert to string if not already
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    Returns a DataFrame with valid emails and count of invalid entries.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    df_valid = df[df[email_column].str.match(email_pattern, na=False)]
    invalid_count = df.shape[0] - df_valid.shape[0]
    
    return df_valid, invalid_count

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'name': ['John Doe', 'Jane Smith', 'John Doe', 'Alice Johnson  '],
#         'email': ['john@example.com', 'invalid-email', 'john@example.com', 'alice@company.org'],
#         'age': [25, 30, 25, 35]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned_df, duplicates_removed = clean_dataframe(df)
#     print(f"\nRemoved {duplicates_removed} duplicate(s)")
#     print("\nCleaned DataFrame:")
#     print(cleaned_df)
#     
#     valid_emails, invalid_count = validate_email_column(cleaned_df, 'email')
#     print(f"\nFound {invalid_count} invalid email(s)")
#     print("\nDataFrame with valid emails:")
#     print(valid_emails)
import pandas as pd
import numpy as np

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
        'count': len(df[column])
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
    all_stats = {}
    
    for column in columns_to_clean:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            all_stats[column] = stats
    
    return cleaned_df, all_stats

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 200],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df, stats = clean_dataset(df)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    print("Summary Statistics:")
    for column, column_stats in stats.items():
        print(f"\n{column}:")
        for stat_name, stat_value in column_stats.items():
            print(f"  {stat_name}: {stat_value}")
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list (list): List containing elements, may include duplicates.
    
    Returns:
        list: New list with duplicates removed, order preserved.
    """
    seen = set()
    result = []
    
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_numeric_data(values, default=0):
    """
    Clean numeric data by converting strings to floats, handling invalid values.
    
    Args:
        values (list): List of values to clean.
        default (float): Default value for invalid entries.
    
    Returns:
        list: Cleaned list of numeric values.
    """
    cleaned = []
    
    for value in values:
        try:
            cleaned.append(float(value))
        except (ValueError, TypeError):
            cleaned.append(default)
    
    return cleaned

def filter_by_threshold(data, threshold, key=None):
    """
    Filter data based on a threshold value.
    
    Args:
        data (list): List of values or dictionaries to filter.
        threshold (float): Threshold value for filtering.
        key (str, optional): If data contains dicts, key to extract value.
    
    Returns:
        list: Filtered data where values >= threshold.
    """
    if key is None:
        return [item for item in data if item >= threshold]
    else:
        return [item for item in data if item.get(key, 0) >= threshold]

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5, 1]
    print("Original:", sample_data)
    print("Cleaned:", remove_duplicates(sample_data))
    
    numeric_data = ["1.5", "2.3", "invalid", "4.7", None]
    print("Numeric original:", numeric_data)
    print("Numeric cleaned:", clean_numeric_data(numeric_data))
    
    scores = [{"name": "A", "score": 85}, {"name": "B", "score": 72}, {"name": "C", "score": 90}]
    print("High scores:", filter_by_threshold(scores, 80, key="score"))
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        return self
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                mask = z_scores < threshold
                valid_indices = df_clean[col].dropna().index[mask]
                df_clean = df_clean.loc[valid_indices.union(df_clean[df_clean[col].isna()].index)]
        
        self.df = df_clean
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val > min_val:
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
            if col in df_filled.columns:
                mean_val = df_filled[col].mean()
                df_filled[col] = df_filled[col].fillna(mean_val)
        
        self.df = df_filled
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns:
                median_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_val)
        
        self.df = df_filled
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'removed_rows': self.get_removed_count(),
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1]
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    data['feature_a'][:50] = np.random.normal(300, 10, 50)
    data['feature_b'][:30] = np.random.normal(500, 20, 30)
    
    df = pd.DataFrame(data)
    
    mask = np.random.random(1000) < 0.05
    df.loc[mask, 'feature_a'] = np.nan
    
    mask = np.random.random(1000) < 0.03
    df.loc[mask, 'feature_b'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    print("\nMissing values:")
    print(sample_df.isnull().sum())
    
    cleaner = DataCleaner(sample_df)
    
    cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    cleaner.fill_missing_mean(['feature_a', 'feature_b'])
    cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    
    cleaned_df = cleaner.get_cleaned_data()
    print("\nCleaned data shape:", cleaned_df.shape)
    print("\nSummary:")
    print(cleaner.get_summary())
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (np.ndarray): Input data array
    column (int): Column index to process
    
    Returns:
    np.ndarray: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    column_data = data[:, column]
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    
    return data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (np.ndarray): Input data array
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation
    """
    if data.size == 0:
        return {"mean": 0, "median": 0, "std": 0}
    
    stats = {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data)
    }
    
    return stats

def validate_data(data):
    """
    Validate data for cleaning operations.
    
    Parameters:
    data (np.ndarray): Input data array
    
    Returns:
    bool: True if data is valid, False otherwise
    """
    if data is None:
        return False
    
    if not isinstance(data, np.ndarray):
        return False
    
    if data.size == 0:
        return False
    
    return True

def process_dataset(data, column_index):
    """
    Main function to process dataset by removing outliers.
    
    Parameters:
    data (np.ndarray): Input data array
    column_index (int): Column index to clean
    
    Returns:
    tuple: (cleaned_data, original_stats, cleaned_stats)
    """
    if not validate_data(data):
        raise ValueError("Invalid input data")
    
    original_stats = calculate_statistics(data[:, column_index])
    
    cleaned_data = remove_outliers_iqr(data, column_index)
    
    if cleaned_data.size > 0:
        cleaned_stats = calculate_statistics(cleaned_data[:, column_index])
    else:
        cleaned_stats = {"mean": 0, "median": 0, "std": 0}
    
    return cleaned_data, original_stats, cleaned_stats
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Cleans a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    cleaned_df = df.copy()

    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {original_shape[0] - cleaned_df.shape[0]} duplicate rows.")

    if fill_missing:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        if fill_missing == 'mean':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        elif fill_missing == 'median':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
        elif fill_missing == 'mode':
            for col in numeric_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
        print(f"Filled missing values in numeric columns using '{fill_missing}' strategy.")

    object_cols = cleaned_df.select_dtypes(include=['object']).columns
    cleaned_df[object_cols] = cleaned_df[object_cols].fillna('Unknown')
    print("Filled missing values in object columns with 'Unknown'.")

    final_shape = cleaned_df.shape
    print(f"Data cleaning complete. Original shape: {original_shape}, Cleaned shape: {final_shape}")

    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, np.nan, 5],
        'B': [10, np.nan, 10, 40, 50],
        'C': ['x', 'y', 'x', 'y', np.nan],
        'D': [100, 200, 100, np.nan, 500]
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaning DataFrame...")
    cleaned = clean_dataframe(df, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned)