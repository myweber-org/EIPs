
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
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
                
        self.df = df_clean.reset_index(drop=True)
        return self
        
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    
        return self
        
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
                    
        return self
        
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].median())
                
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
            'columns': list(self.df.columns),
            'dtypes': dict(self.df.dtypes)
        }
        return summary

def process_dataset(filepath, output_path=None):
    try:
        df = pd.read_csv(filepath)
        cleaner = DataCleaner(df)
        
        cleaner.fill_missing_median()
        cleaner.remove_outliers_iqr()
        cleaner.normalize_minmax()
        
        summary = cleaner.get_summary()
        cleaned_df = cleaner.get_cleaned_data()
        
        if output_path:
            cleaned_df.to_csv(output_path, index=False)
            
        return cleaned_df, summary
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None, None
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def remove_outliers_zscore(dataframe, column, threshold=3):
    z_scores = np.abs(stats.zscore(dataframe[column]))
    return dataframe[z_scores < threshold]

def normalize_minmax(dataframe, column):
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    dataframe[column + '_normalized'] = (dataframe[column] - min_val) / (max_val - min_val)
    return dataframe

def normalize_zscore(dataframe, column):
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    dataframe[column + '_standardized'] = (dataframe[column] - mean_val) / std_val
    return dataframe

def handle_missing_values(dataframe, column, strategy='mean'):
    if strategy == 'mean':
        fill_value = dataframe[column].mean()
    elif strategy == 'median':
        fill_value = dataframe[column].median()
    elif strategy == 'mode':
        fill_value = dataframe[column].mode()[0]
    else:
        fill_value = 0
    
    dataframe[column] = dataframe[column].fillna(fill_value)
    return dataframe

def validate_dataframe(dataframe, required_columns):
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column] = df[column].fillna(df[column].median())
    
    # Remove outliers using z-score
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    for column in numeric_cols:
        if df[column].std() != 0:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization='minmax'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_removal:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        
        if normalization == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalization == 'zscore':
            cleaned_df = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_dataframe(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True

def get_summary_statistics(df, numeric_columns):
    summary = {}
    for col in numeric_columns:
        if col in df.columns:
            summary[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'count': df[col].count(),
                'missing': df[col].isnull().sum()
            }
    return pd.DataFrame(summary).T

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 200, 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("\nOriginal summary statistics:")
    print(get_summary_statistics(sample_data, ['feature1', 'feature2', 'feature3']))
    
    cleaned_data = clean_dataset(
        sample_data, 
        ['feature1', 'feature2', 'feature3'],
        outlier_removal=True,
        normalization='zscore'
    )
    
    print("\nCleaned data shape:", cleaned_data.shape)
    print("\nCleaned summary statistics:")
    print(get_summary_statistics(cleaned_data, ['feature1', 'feature2', 'feature3']))import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers from a pandas Series using the IQR method.
    """
    if not isinstance(data, pd.Series):
        series = pd.Series(data)
    else:
        series = data.copy()
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_series = series[(series >= lower_bound) & (series <= upper_bound)]
    return filtered_series

def normalize_minmax(data):
    """
    Normalize data to [0, 1] range using min-max scaling.
    """
    data_array = np.array(data)
    if len(data_array) == 0:
        return data_array
    
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    
    if max_val == min_val:
        return np.zeros_like(data_array)
    
    normalized = (data_array - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data):
    """
    Standardize data to have zero mean and unit variance.
    """
    data_array = np.array(data)
    if len(data_array) == 0:
        return data_array
    
    mean_val = np.mean(data_array)
    std_val = np.std(data_array)
    
    if std_val == 0:
        return np.zeros_like(data_array)
    
    standardized = (data_array - mean_val) / std_val
    return standardized

def clean_dataframe(df, numeric_columns=None, outlier_threshold=1.5):
    """
    Clean a DataFrame by removing outliers and normalizing numeric columns.
    """
    cleaned_df = df.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            original_data = cleaned_df[col].dropna()
            if len(original_data) > 0:
                cleaned_series = remove_outliers_iqr(original_data, col, outlier_threshold)
                cleaned_df.loc[cleaned_df[col].notna(), col] = cleaned_series.reindex(cleaned_df[cleaned_df[col].notna()].index)
    
    return cleaned_df

def process_dataset(file_path, output_path=None):
    """
    Load, clean, and optionally save a dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with shape: {df.shape}")
        
        cleaned_df = clean_dataframe(df)
        print(f"Cleaned dataset shape: {cleaned_df.shape}")
        
        if output_path:
            cleaned_df.to_csv(output_path, index=False)
            print(f"Saved cleaned data to: {output_path}")
        
        return cleaned_df
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None

if __name__ == "__main__":
    sample_data = np.random.normal(100, 15, 1000)
    sample_data_with_outliers = np.append(sample_data, [500, -200, 1000])
    
    print("Original data shape:", sample_data_with_outliers.shape)
    cleaned = remove_outliers_iqr(sample_data_with_outliers, 'sample', threshold=1.5)
    print("Cleaned data shape:", cleaned.shape)
    
    normalized = normalize_minmax(cleaned)
    print("Normalized data range: [{:.3f}, {:.3f}]".format(np.min(normalized), np.max(normalized)))
    
    standardized = standardize_zscore(cleaned)
    print("Standardized data stats - Mean: {:.3f}, Std: {:.3f}".format(np.mean(standardized), np.std(standardized)))
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np
import pandas as pd
from scipy import stats

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
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_method: 'iqr' or 'zscore' (default: 'iqr')
        normalize_method: 'minmax' or 'zscore' (default: 'minmax')
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    # Remove outliers
    for column in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
    
    # Normalize data
    for column in numeric_columns:
        if normalize_method == 'minmax':
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[column] = normalize_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown normalize method: {normalize_method}")
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
        allow_nan: whether to allow NaN values
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if not allow_nan and data.isnull().any().any():
        return False, "Dataset contains NaN values"
    
    return True, "Dataset is valid"
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    dataframe (pd.DataFrame): Input dataframe
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def calculate_basic_stats(dataframe, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    dataframe (pd.DataFrame): Input dataframe
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistics
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    stats = {
        'mean': dataframe[column].mean(),
        'median': dataframe[column].median(),
        'std': dataframe[column].std(),
        'min': dataframe[column].min(),
        'max': dataframe[column].max(),
        'count': dataframe[column].count(),
        'missing': dataframe[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(dataframe, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input dataframe
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = dataframe.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not process column '{column}': {e}")
    
    return cleaned_df

def generate_cleaning_report(original_df, cleaned_df):
    """
    Generate a report comparing original and cleaned data.
    
    Parameters:
    original_df (pd.DataFrame): Original dataframe
    cleaned_df (pd.DataFrame): Cleaned dataframe
    
    Returns:
    pd.DataFrame: Report dataframe
    """
    report_data = []
    
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns
    
    for column in numeric_cols:
        if column in original_df.columns and column in cleaned_df.columns:
            original_stats = calculate_basic_stats(original_df, column)
            cleaned_stats = calculate_basic_stats(cleaned_df, column)
            
            report_data.append({
                'column': column,
                'original_count': original_stats['count'],
                'cleaned_count': cleaned_stats['count'],
                'removed_count': original_stats['count'] - cleaned_stats['count'],
                'removed_percentage': ((original_stats['count'] - cleaned_stats['count']) / original_stats['count']) * 100,
                'original_mean': original_stats['mean'],
                'cleaned_mean': cleaned_stats['mean'],
                'original_std': original_stats['std'],
                'cleaned_std': cleaned_stats['std']
            })
    
    report_df = pd.DataFrame(report_data)
    return report_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample data with outliers
    sample_data = {
        'temperature': np.concatenate([np.random.normal(20, 5, 90), [100, -10, 150]]),
        'humidity': np.concatenate([np.random.normal(50, 10, 90), [200, -5, 300]]),
        'pressure': np.random.normal(1013, 10, 93),
        'category': ['A', 'B', 'C'] * 31
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data shape:", df.shape)
    
    # Clean the data
    cleaned_df = clean_numeric_data(df, ['temperature', 'humidity'])
    print("Cleaned data shape:", cleaned_df.shape)
    
    # Generate report
    report = generate_cleaning_report(df, cleaned_df)
    print("\nCleaning Report:")
    print(report.to_string())
import pandas as pd
import numpy as np

def clean_dataframe(df, column_mapping=None, drop_duplicates=True):
    """
    Clean a pandas DataFrame by standardizing column names and removing duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict): Optional dictionary mapping old column names to new ones
        drop_duplicates (bool): Whether to remove duplicate rows
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Standardize column names
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Convert column names to lowercase and replace spaces with underscores
    cleaned_df.columns = cleaned_df.columns.str.lower().str.replace(' ', '_')
    
    # Remove duplicate rows if specified
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Replace NaN values with None for consistency
    cleaned_df = cleaned_df.replace({np.nan: None})
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def process_csv_file(input_path, output_path, **kwargs):
    """
    Process a CSV file through the data cleaning pipeline.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save cleaned CSV file
        **kwargs: Additional arguments passed to clean_dataframe
    
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    try:
        # Read input CSV
        df = pd.read_csv(input_path)
        
        # Clean the data
        cleaned_df = clean_dataframe(df, **kwargs)
        
        # Validate the cleaned data
        is_valid, message = validate_dataframe(cleaned_df)
        
        if not is_valid:
            print(f"Validation failed: {message}")
            return False
        
        # Save cleaned data
        cleaned_df.to_csv(output_path, index=False)
        print(f"Successfully cleaned data saved to: {output_path}")
        print(f"Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
        
        return True
        
    except FileNotFoundError:
        print(f"Input file not found: {input_path}")
        return False
    except pd.errors.EmptyDataError:
        print(f"Input file is empty: {input_path}")
        return False
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Customer ID': [1, 2, 2, 3, 4],
        'First Name': ['John', 'Jane', 'Jane', 'Bob', 'Alice'],
        'Last Name': ['Doe', 'Smith', 'Smith', 'Johnson', 'Brown'],
        'Email': ['john@example.com', 'jane@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com'],
        'Purchase Amount': [100.50, 200.75, 200.75, 150.00, 300.25]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    column_mapping = {'Customer ID': 'customer_id', 'Purchase Amount': 'amount'}
    cleaned_df = clean_dataframe(df, column_mapping=column_mapping)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Args:
        data (pd.DataFrame): The input DataFrame.
        column (str): The column name to process.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a specified column.
    
    Args:
        data (pd.DataFrame): The input DataFrame.
        column (str): The column name to analyze.
    
    Returns:
        dict: A dictionary containing count, mean, std, min, max.
    """
    stats = {
        'count': data[column].count(),
        'mean': data[column].mean(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max()
    }
    return stats
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: dictionary to rename columns
        drop_duplicates: whether to remove duplicate rows
        normalize_text: whether to normalize text columns
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
    
    return cleaned_df

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for required columns and minimum row count.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
        min_rows: minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if len(df) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame validation passed"

def sample_dataframe(df, n=5, random_state=42):
    """
    Return a sample of the DataFrame for inspection.
    
    Args:
        df: pandas DataFrame to sample
        n: number of rows to sample
        random_state: random seed for reproducibility
    
    Returns:
        Sampled pandas DataFrame
    """
    if len(df) <= n:
        return df
    
    return df.sample(n=n, random_state=random_state)
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
                df_clean = df_clean[(z_scores < threshold) | df_clean[col].isna()]
        
        removed_count = len(self.df) - len(df_clean)
        self.df = df_clean
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                col_min = df_normalized[col].min()
                col_max = df_normalized[col].max()
                if col_max > col_min:
                    df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
        
        self.df = df_normalized
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                col_mean = df_normalized[col].mean()
                col_std = df_normalized[col].std()
                if col_std > 0:
                    df_normalized[col] = (df_normalized[col] - col_mean) / col_std
        
        self.df = df_normalized
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removal_stats(self):
        final_rows = len(self.df)
        original_rows = self.original_shape[0]
        removed_rows = original_rows - final_rows
        removal_percentage = (removed_rows / original_rows * 100) if original_rows > 0 else 0
        
        return {
            'original_rows': original_rows,
            'final_rows': final_rows,
            'removed_rows': removed_rows,
            'removal_percentage': removal_percentage
        }
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a specific column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]