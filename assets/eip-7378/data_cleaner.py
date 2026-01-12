import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or array-like): The dataset containing the column to clean.
    column (int or str): The index or name of the column to process.
    
    Returns:
    numpy.ndarray: Data with outliers removed from the specified column.
    """
    if isinstance(data, list):
        data = np.array(data)
    
    column_data = data[:, column] if data.ndim > 1 else data
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    
    if data.ndim > 1:
        return data[mask]
    else:
        return data[mask]

def calculate_basic_stats(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (array-like): Input data.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    if len(data) == 0:
        return {"mean": 0, "median": 0, "std": 0}
    
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data)
    }

if __name__ == "__main__":
    sample_data = np.random.randn(1000) * 10 + 50
    sample_data[50] = 200
    sample_data[150] = -100
    
    print("Original data shape:", sample_data.shape)
    print("Original stats:", calculate_basic_stats(sample_data))
    
    cleaned_data = remove_outliers_iqr(sample_data, 0)
    
    print("Cleaned data shape:", cleaned_data.shape)
    print("Cleaned stats:", calculate_basic_stats(cleaned_data))