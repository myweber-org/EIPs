
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or np.array): The dataset.
    column (int): Index of the column to process.
    
    Returns:
    np.array: Data with outliers removed.
    """
    data = np.array(data)
    col_data = data[:, column].astype(float)
    
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    cleaned_data = data[mask]
    
    return cleaned_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    data (list or np.array): The dataset.
    column (int): Index of the column to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    data = np.array(data)
    col_data = data[:, column].astype(float)
    
    stats = {
        'mean': np.mean(col_data),
        'median': np.median(col_data),
        'std': np.std(col_data)
    }
    
    return stats

if __name__ == "__main__":
    sample_data = [
        [1, 150.5],
        [2, 165.3],
        [3, 172.1],
        [4, 158.7],
        [5, 210.8],
        [6, 155.2],
        [7, 168.9],
        [8, 300.2],
        [9, 162.4],
        [10, 175.6]
    ]
    
    print("Original data:")
    print(sample_data)
    
    cleaned = remove_outliers_iqr(sample_data, 1)
    print("\nCleaned data:")
    print(cleaned)
    
    stats = calculate_statistics(cleaned, 1)
    print("\nStatistics for cleaned data:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")