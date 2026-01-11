import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or array-like): The dataset
    column (int or str): Column index or name if using pandas DataFrame
    
    Returns:
    tuple: (cleaned_data, outliers_removed)
    """
    if isinstance(data, list):
        data_array = np.array(data)
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        mask = (data_array >= lower_bound) & (data_array <= upper_bound)
        cleaned = data_array[mask]
        outliers = data_array[~mask]
        
        return cleaned.tolist(), outliers.tolist()
    
    elif hasattr(data, 'iloc'):  # pandas DataFrame
        series = data[column] if isinstance(column, str) else data.iloc[:, column]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        mask = (series >= lower_bound) & (series <= upper_bound)
        cleaned_data = data[mask].copy()
        outliers_data = data[~mask].copy()
        
        return cleaned_data, outliers_data
    
    else:
        raise TypeError("Unsupported data type. Use list or pandas DataFrame.")

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (array-like): Input data
    
    Returns:
    dict: Dictionary containing statistics
    """
    data_array = np.array(data)
    stats = {
        'mean': np.mean(data_array),
        'median': np.median(data_array),
        'std': np.std(data_array),
        'min': np.min(data_array),
        'max': np.max(data_array),
        'count': len(data_array)
    }
    return stats

if __name__ == "__main__":
    # Example usage
    sample_data = [10, 12, 13, 15, 16, 18, 20, 22, 24, 100]  # 100 is an outlier
    
    print("Original data:", sample_data)
    print("Statistics:", calculate_statistics(sample_data))
    
    cleaned, outliers = remove_outliers_iqr(sample_data, 0)
    print("Cleaned data:", cleaned)
    print("Outliers removed:", outliers)
    print("Cleaned statistics:", calculate_statistics(cleaned))