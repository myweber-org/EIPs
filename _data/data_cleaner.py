
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list: A list containing elements (must be hashable).
    
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

def clean_data_with_counts(data):
    """
    Clean data and return both cleaned data and duplicate count.
    
    Args:
        data: List of items to clean.
    
    Returns:
        Tuple of (cleaned_list, duplicate_count)
    """
    original_len = len(data)
    cleaned = remove_duplicates(data)
    duplicates_removed = original_len - len(cleaned)
    return cleaned, duplicates_removed

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 4, 5, 1, 6]
    cleaned_data, count = clean_data_with_counts(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned_data}")
    print(f"Duplicates removed: {count}")