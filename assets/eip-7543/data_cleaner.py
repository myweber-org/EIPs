
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
    if data.size == 0:
        return data
    
    col_data = data[:, column]
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (np.ndarray): Input data array
    column (int): Column index to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if data.size == 0:
        return {}
    
    col_data = data[:, column]
    stats = {
        'mean': np.mean(col_data),
        'median': np.median(col_data),
        'std': np.std(col_data),
        'min': np.min(col_data),
        'max': np.max(col_data)
    }
    return stats

def clean_dataset(data, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Parameters:
    data (np.ndarray): Input data array
    columns_to_clean (list): List of column indices to clean
    
    Returns:
    np.ndarray: Cleaned data array
    """
    cleaned_data = data.copy()
    
    for column in columns_to_clean:
        if column < cleaned_data.shape[1]:
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    return cleaned_data

if __name__ == "__main__":
    # Example usage
    sample_data = np.random.randn(100, 3) * 10 + 50
    sample_data[0, 0] = 200  # Add an outlier
    
    print("Original data shape:", sample_data.shape)
    print("Original statistics:", calculate_statistics(sample_data, 0))
    
    cleaned = clean_dataset(sample_data, [0, 1, 2])
    print("Cleaned data shape:", cleaned.shape)
    print("Cleaned statistics:", calculate_statistics(cleaned, 0))