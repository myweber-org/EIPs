
def remove_duplicates(input_list):
    """
    Remove duplicate items from a list while preserving order.
    
    Args:
        input_list: A list that may contain duplicate elements.
    
    Returns:
        A new list with duplicates removed.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data_with_threshold(data, threshold=None):
    """
    Clean data by removing duplicates, optionally filtering by frequency threshold.
    
    Args:
        data: List of items to clean.
        threshold: Minimum frequency for items to keep (inclusive).
    
    Returns:
        Cleaned list and frequency statistics.
    """
    from collections import Counter
    
    counter = Counter(data)
    
    if threshold is not None:
        filtered_items = [item for item, count in counter.items() if count >= threshold]
        cleaned_data = remove_duplicates([item for item in data if item in filtered_items])
    else:
        cleaned_data = remove_duplicates(data)
    
    return cleaned_data, dict(counter)

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 4, 5, 1, 6]
    
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
    
    cleaned_with_threshold, stats = clean_data_with_threshold(sample_data, threshold=2)
    print(f"Threshold cleaned: {cleaned_with_threshold}")
    print(f"Frequency stats: {stats}")