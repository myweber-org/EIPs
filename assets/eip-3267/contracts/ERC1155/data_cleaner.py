
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (str): Method to fill missing values - 'mean', 'median', or 'drop'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate that DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if df.empty:
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None],
        'B': [4, None, 6, 7, 8],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nData validation result: {is_valid}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Column index to process
    
    Returns:
    numpy.ndarray: Data with outliers removed
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

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Column index to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if data.size == 0:
        return {}
    
    column_data = data[:, column]
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data),
        'min': np.min(column_data),
        'max': np.max(column_data),
        'count': len(column_data)
    }
    
    return stats

def process_dataset(data, columns_to_clean):
    """
    Process multiple columns for outlier removal and return cleaned dataset.
    
    Parameters:
    data (numpy.ndarray): Input data array
    columns_to_clean (list): List of column indices to clean
    
    Returns:
    numpy.ndarray: Cleaned dataset
    dict: Dictionary of statistics for each cleaned column
    """
    if not isinstance(columns_to_clean, list):
        columns_to_clean = [columns_to_clean]
    
    cleaned_data = data.copy()
    all_stats = {}
    
    for column in columns_to_clean:
        try:
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
            stats = calculate_basic_stats(cleaned_data, column)
            all_stats[f'column_{column}'] = stats
        except Exception as e:
            print(f"Error processing column {column}: {str(e)}")
            continue
    
    return cleaned_data, all_stats

def validate_data_shape(data, expected_columns):
    """
    Validate that data has the expected number of columns.
    
    Parameters:
    data (numpy.ndarray): Input data array
    expected_columns (int): Expected number of columns
    
    Returns:
    bool: True if shape is valid, False otherwise
    """
    if len(data.shape) != 2:
        return False
    
    return data.shape[1] == expected_columns

def example_usage():
    """
    Example demonstrating how to use the data cleaning functions.
    """
    np.random.seed(42)
    
    sample_data = np.random.randn(1000, 3)
    sample_data[:, 1] = sample_data[:, 1] * 10 + 50
    
    columns_to_process = [0, 1]
    
    if validate_data_shape(sample_data, 3):
        cleaned_data, statistics = process_dataset(sample_data, columns_to_process)
        
        print(f"Original data shape: {sample_data.shape}")
        print(f"Cleaned data shape: {cleaned_data.shape}")
        
        for col, stats in statistics.items():
            print(f"\nStatistics for {col}:")
            for key, value in stats.items():
                print(f"  {key}: {value:.4f}")
    else:
        print("Invalid data shape")

if __name__ == "__main__":
    example_usage()