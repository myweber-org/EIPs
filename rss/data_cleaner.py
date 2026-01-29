
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    unique_list = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def clean_data(data):
    """
    Main cleaning function that processes the input data.
    Handles None values and non-list inputs gracefully.
    """
    if data is None:
        return []
    
    if not isinstance(data, list):
        try:
            data = list(data)
        except TypeError:
            return []
    
    return remove_duplicates(data)

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5, 1, 6]
    cleaned = clean_data(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")