
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def normalize_minmax(data, column):
    """
    Normalize column using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by handling missing values and outliers
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    stats = {}
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df, outliers = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
            
            stats[col] = {
                'outliers_removed': outliers,
                'outlier_percentage': (outliers / original_len) * 100 if original_len > 0 else 0
            }
    
    return cleaned_df, stats

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def generate_summary(df):
    """
    Generate summary statistics for DataFrame
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.randint(1, 100, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame shape:", df.shape)
    
    cleaned_df, stats = clean_dataset(df)
    print("Cleaned DataFrame shape:", cleaned_df.shape)
    
    for col, stat in stats.items():
        print(f"{col}: Removed {stat['outliers_removed']} outliers ({stat['outlier_percentage']:.2f}%)")
    
    summary = generate_summary(cleaned_df)
    print(f"\nSummary: {summary['total_rows']} rows, {summary['total_columns']} columns")
import pandas as pd
import numpy as np

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, fill_na=True):
    """
    Clean a pandas DataFrame by standardizing columns, removing duplicates,
    and handling missing values.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_na:
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def validate_dataframe(df, required_columns=None, unique_columns=None):
    """
    Validate DataFrame structure and content.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if unique_columns:
        for col in unique_columns:
            if col in df.columns and df[col].duplicated().any():
                raise ValueError(f"Column '{col}' contains duplicate values")
    
    return True

def process_csv_file(input_path, output_path, **kwargs):
    """
    Read CSV file, clean data, and save to output path.
    """
    try:
        df = pd.read_csv(input_path)
        cleaned_df = clean_dataframe(df, **kwargs)
        
        if validate_dataframe(cleaned_df):
            cleaned_df.to_csv(output_path, index=False)
            return True, f"Data cleaned successfully. Saved to {output_path}"
    except Exception as e:
        return False, f"Error processing file: {str(e)}"
    
    return False, "Unknown error occurred"

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', None],
        'Age': [25, 30, 25, None, 35],
        'City': ['NYC', 'LA', 'NYC', 'Chicago', 'Boston']
    })
    
    cleaned = clean_dataframe(sample_data)
    print("Original data:")
    print(sample_data)
    print("\nCleaned data:")
    print(cleaned)
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
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

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
    
    return (data[column] - min_val) / (max_val - min_val)

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
    
    return (data[column] - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='zscore'):
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
    Validate data structure and content
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan and data.isnull().any().any():
        nan_columns = data.columns[data.isnull().any()].tolist()
        raise ValueError(f"Data contains NaN values in columns: {nan_columns}")
    
    return True
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
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

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling to range [0, 1].
    
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
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names (if None, auto-detect)
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            # Remove outliers
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
            
            # Normalize the column
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
    
    return cleaned_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        data: pandas DataFrame
        column: column name
    
    Returns:
        Dictionary with statistics
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'q1': data[column].quantile(0.25),
        'q3': data[column].quantile(0.75),
        'count': data[column].count(),
        'missing': data[column].isnull().sum()
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("\nOriginal statistics for feature_a:")
    print(calculate_statistics(sample_data, 'feature_a'))
    
    cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b'])
    print("\nCleaned data shape:", cleaned.shape)
    print("\nCleaned statistics for feature_a:")
    print(calculate_statistics(cleaned, 'feature_a'))
def clean_data(data):
    """
    Remove duplicate entries from a list and sort the remaining items.
    """
    if not isinstance(data, list):
        raise TypeError("Input must be a list")
    
    unique_data = list(set(data))
    unique_data.sort()
    return unique_data
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a dataset by handling missing values and outliers.
    
    Args:
        df (pd.DataFrame): Input dataframe
        missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        for col in numeric_cols:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    elif missing_strategy == 'median':
        for col in numeric_cols:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    elif missing_strategy == 'mode':
        for col in numeric_cols:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    elif missing_strategy == 'drop':
        df_clean.dropna(subset=numeric_cols, inplace=True)
    
    # Handle outliers using Z-score method
    for col in numeric_cols:
        z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
        df_clean = df_clean[z_scores < outlier_threshold]
    
    # Reset index after cleaning
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "Dataframe is empty"
    
    if len(df) < min_rows:
        return False, f"Dataframe has less than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    
    # Clean the data
    cleaned_df = clean_dataset(df, missing_strategy='mean', outlier_threshold=2)
    print("\nCleaned data:")
    print(cleaned_df)
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned_df, required_columns=['A', 'B', 'C'], min_rows=2)
    print(f"\nValidation: {is_valid} - {message}")