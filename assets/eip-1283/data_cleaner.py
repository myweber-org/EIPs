
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or np.array): The dataset.
    column (int): Index of the column to clean.
    
    Returns:
    np.array: Data with outliers removed.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    column_data = data[:, column].astype(float)
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    
    return data[mask]

def clean_dataset(data, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Parameters:
    data (list or np.array): The dataset.
    columns_to_clean (list): List of column indices to clean.
    
    Returns:
    np.array: Cleaned dataset.
    """
    cleaned_data = np.array(data)
    
    for column in columns_to_clean:
        cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    return cleaned_data

if __name__ == "__main__":
    sample_data = [
        [1, 150.5, 30],
        [2, 160.2, 35],
        [3, 170.8, 40],
        [4, 180.1, 45],
        [5, 190.5, 50],
        [6, 200.0, 55],
        [7, 250.0, 60],
        [8, 300.0, 65],
        [9, 350.0, 70],
        [10, 400.0, 75]
    ]
    
    print("Original data:")
    for row in sample_data:
        print(row)
    
    cleaned = clean_dataset(sample_data, [1, 2])
    
    print("\nCleaned data:")
    for row in cleaned:
        print(row)