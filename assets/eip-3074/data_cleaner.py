
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list: List of elements (must be hashable)
    
    Returns:
        List with duplicates removed
    """
    seen = set()
    result = []
    
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result


def clean_numeric_data(values, default=0):
    """
    Clean numeric data by converting strings to floats and handling invalid values.
    
    Args:
        values: List of values to clean
        default: Default value for invalid entries
    
    Returns:
        List of cleaned numeric values
    """
    cleaned = []
    
    for value in values:
        try:
            # Try to convert to float
            num = float(value)
            cleaned.append(num)
        except (ValueError, TypeError):
            # Use default for invalid values
            cleaned.append(default)
    
    return cleaned


def filter_by_threshold(data, threshold, key=None):
    """
    Filter data based on a threshold value.
    
    Args:
        data: List of values or dictionaries
        threshold: Minimum value to include
        key: If data contains dictionaries, key to extract value from
    
    Returns:
        Filtered list
    """
    if key is None:
        # Simple list of values
        return [x for x in data if x >= threshold]
    else:
        # List of dictionaries
        return [item for item in data if item.get(key, 0) >= threshold]


# Example usage
if __name__ == "__main__":
    # Test remove_duplicates
    sample_data = [1, 2, 2, 3, 4, 4, 4, 5]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
    
    # Test clean_numeric_data
    mixed_data = ["1.5", "2.7", "invalid", "3.0", None, "4.2"]
    numeric_data = clean_numeric_data(mixed_data)
    print(f"\nMixed data: {mixed_data}")
    print(f"Numeric data: {numeric_data}")
    
    # Test filter_by_threshold
    scores = [45, 78, 92, 33, 67, 88]
    high_scores = filter_by_threshold(scores, 70)
    print(f"\nAll scores: {scores}")
    print(f"High scores (>=70): {high_scores}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    Returns a filtered DataFrame.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_stats(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max()
    }
    return stats

if __name__ == "__main__":
    import pandas as pd
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'values': np.random.normal(100, 15, 1000)
    })
    print("Original data shape:", sample_data.shape)
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print("Cleaned data shape:", cleaned_data.shape)
    stats = calculate_summary_stats(cleaned_data, 'values')
    print("Summary statistics:", stats)