
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
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
    Clean data by removing duplicates, optionally filtering by frequency threshold.
    
    Args:
        data_list (list): List of data items to clean.
        threshold (int, optional): Minimum frequency for items to keep.
    
    Returns:
        list: Cleaned list of unique items.
    """
    from collections import Counter
    
    if not data_list:
        return []
    
    # Count frequencies
    counter = Counter(data_list)
    
    if threshold is not None:
        # Filter items by frequency threshold
        filtered_items = [item for item, count in counter.items() if count >= threshold]
        return remove_duplicates(filtered_items)
    else:
        # Just remove duplicates
        return remove_duplicates(data_list)

def validate_input(data):
    """
    Validate that input is a list or convertible to list.
    
    Args:
        data: Input data to validate.
    
    Returns:
        list: Validated list.
    
    Raises:
        TypeError: If input cannot be converted to list.
    """
    if isinstance(data, list):
        return data
    elif hasattr(data, '__iter__'):
        return list(data)
    else:
        raise TypeError("Input must be iterable")

# Example usage
if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 4, 5]
    print(f"Original data: {sample_data}")
    print(f"Cleaned data: {remove_duplicates(sample_data)}")
    print(f"Cleaned with threshold 2: {clean_data_with_threshold(sample_data, threshold=2)}")