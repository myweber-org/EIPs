
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or np.array): Input data
    column (int): Column index for 2D data, or None for 1D data
    
    Returns:
    np.array: Data with outliers removed
    """
    if column is not None:
        column_data = data[:, column]
    else:
        column_data = np.array(data)
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    if column is not None:
        mask = (data[:, column] >= lower_bound) & (data[:, column] <= upper_bound)
        return data[mask]
    else:
        mask = (column_data >= lower_bound) & (column_data <= upper_bound)
        return column_data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (np.array): Input data
    
    Returns:
    dict: Dictionary containing mean, median, std, min, max
    """
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }

def clean_dataset(data, column=None):
    """
    Main function to clean dataset by removing outliers and returning statistics.
    
    Parameters:
    data (list or np.array): Input data
    column (int): Column index for 2D data, or None for 1D data
    
    Returns:
    tuple: (cleaned_data, original_stats, cleaned_stats)
    """
    original_stats = calculate_statistics(data if column is None else data[:, column])
    
    cleaned_data = remove_outliers_iqr(data, column)
    
    if column is not None:
        cleaned_stats = calculate_statistics(cleaned_data[:, column])
    else:
        cleaned_stats = calculate_statistics(cleaned_data)
    
    return cleaned_data, original_stats, cleaned_stats

if __name__ == "__main__":
    # Example usage
    sample_data = np.random.randn(1000, 3) * 10 + 50
    sample_data[0:5, 1] = 200  # Add some outliers
    
    cleaned, orig_stats, clean_stats = clean_dataset(sample_data, column=1)
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Cleaned data shape: {cleaned.shape}")
    print(f"Original stats: {orig_stats}")
    print(f"Cleaned stats: {clean_stats}")
def remove_duplicates(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result