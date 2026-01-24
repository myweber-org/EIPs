import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or array): The dataset containing the column.
    column (int or str): The index or name of the column to clean.
    
    Returns:
    numpy.ndarray: Data with outliers removed.
    """
    data_array = np.array(data)
    col_data = data_array[:, column] if isinstance(column, int) else data_array[column]
    
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    cleaned_data = data_array[mask]
    
    return cleaned_data

def example_usage():
    sample_data = np.array([
        [1, 150],
        [2, 200],
        [3, 250],
        [4, 300],
        [5, 1000],
        [6, 50]
    ])
    
    print("Original data:")
    print(sample_data)
    
    cleaned = remove_outliers_iqr(sample_data, column=1)
    print("\nCleaned data (outliers removed from column 1):")
    print(cleaned)

if __name__ == "__main__":
    example_usage()