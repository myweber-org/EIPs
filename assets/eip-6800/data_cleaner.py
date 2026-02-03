
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or array-like): The dataset containing the column to clean.
    column (int or str): Index or name of the column to process.
    
    Returns:
    tuple: (cleaned_data, outlier_indices)
    """
    if isinstance(data, list):
        data_array = np.array(data)
    else:
        data_array = np.asarray(data)
    
    if isinstance(column, str):
        raise ValueError("Column names not supported with array input. Use integer index.")
    
    column_data = data_array[:, column].astype(float)
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (column_data < lower_bound) | (column_data > upper_bound)
    outlier_indices = np.where(outlier_mask)[0]
    
    cleaned_data = data_array[~outlier_mask]
    
    return cleaned_data, outlier_indices

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    data (array-like): The dataset.
    column (int): Index of the column.
    
    Returns:
    dict: Dictionary containing statistics.
    """
    column_data = data[:, column].astype(float)
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data),
        'min': np.min(column_data),
        'max': np.max(column_data),
        'q1': np.percentile(column_data, 25),
        'q3': np.percentile(column_data, 75)
    }
    
    return stats

if __name__ == "__main__":
    sample_data = np.array([
        [1, 150.5],
        [2, 165.2],
        [3, 172.8],
        [4, 158.1],
        [5, 210.0],
        [6, 155.3],
        [7, 300.0],
        [8, 162.7],
        [9, 168.9],
        [10, 152.4]
    ])
    
    print("Original data shape:", sample_data.shape)
    print("Original data:")
    print(sample_data)
    
    cleaned, outliers = remove_outliers_iqr(sample_data, 1)
    
    print("\nOutlier indices:", outliers)
    print("Cleaned data shape:", cleaned.shape)
    print("Cleaned data:")
    print(cleaned)
    
    stats = calculate_basic_stats(sample_data, 1)
    print("\nStatistics for column 1:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")