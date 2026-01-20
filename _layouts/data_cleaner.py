def remove_duplicates(input_list):
    """
    Remove duplicate items from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data(data):
    """
    Main cleaning function that processes a list of data.
    """
    if not isinstance(data, list):
        raise TypeError("Input must be a list")
    
    cleaned = remove_duplicates(data)
    return cleaned

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5, 1, 6]
    result = clean_data(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {result}")