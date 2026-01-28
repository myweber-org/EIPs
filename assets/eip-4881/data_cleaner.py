import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
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
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Column index to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    cleaned_data = remove_outliers_iqr(data, column)
    column_data = cleaned_data[:, column]
    
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
    Process multiple columns for outlier removal and statistics.
    
    Parameters:
    data (numpy.ndarray): Input data array
    columns_to_clean (list): List of column indices to process
    
    Returns:
    tuple: (cleaned_data, statistics_dict)
    """
    if not columns_to_clean:
        return data, {}
    
    cleaned_data = data.copy()
    all_stats = {}
    
    for col in columns_to_clean:
        cleaned_data = remove_outliers_iqr(cleaned_data, col)
        all_stats[col] = calculate_statistics(cleaned_data, col)
    
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
    if data.ndim != 2:
        return False
    
    return data.shape[1] == expected_columns

def example_usage():
    """
    Example demonstrating how to use the data cleaning functions.
    """
    np.random.seed(42)
    sample_data = np.random.randn(100, 3) * 10 + 50
    
    print("Original data shape:", sample_data.shape)
    
    cleaned_data, stats = process_dataset(sample_data, [0, 1, 2])
    
    print("Cleaned data shape:", cleaned_data.shape)
    
    for col, col_stats in stats.items():
        print(f"\nColumn {col} statistics:")
        for key, value in col_stats.items():
            print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    example_usage()