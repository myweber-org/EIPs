
def remove_duplicates(input_list):
    """
    Remove duplicate items from a list while preserving the original order.
    
    Args:
        input_list (list): The list from which duplicates should be removed.
    
    Returns:
        list: A new list with duplicates removed.
    """
    seen = set()
    result = []
    
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_data_with_threshold(data_list, threshold=None):
    """
    Clean data by removing duplicates, optionally filtering by occurrence threshold.
    
    Args:
        data_list (list): List of data items to clean.
        threshold (int, optional): Minimum occurrences to keep an item. Defaults to None.
    
    Returns:
        list: Cleaned list of data items.
    """
    if not data_list:
        return []
    
    # First remove duplicates while preserving order
    unique_data = remove_duplicates(data_list)
    
    # Apply threshold filter if specified
    if threshold is not None and threshold > 0:
        from collections import Counter
        counts = Counter(data_list)
        unique_data = [item for item in unique_data if counts[item] >= threshold]
    
    return unique_data

def validate_input_data(data):
    """
    Validate that input is a list or convertible to list.
    
    Args:
        data: Input data to validate.
    
    Returns:
        list: Validated list data.
    
    Raises:
        TypeError: If data cannot be converted to a list.
    """
    if isinstance(data, list):
        return data
    elif hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
        return list(data)
    else:
        raise TypeError("Input must be an iterable or list")

# Example usage (commented out for production)
if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 4, 5, 1]
    print("Original:", sample_data)
    print("Cleaned:", remove_duplicates(sample_data))
    print("Threshold 2:", clean_data_with_threshold(sample_data, threshold=2))