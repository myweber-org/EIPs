
def clean_data(data):
    """
    Remove duplicates from a list and sort the remaining items.
    
    Args:
        data (list): A list of items (must be comparable).
    
    Returns:
        list: A sorted list with duplicates removed.
    """
    if not isinstance(data, list):
        raise TypeError("Input must be a list")
    
    unique_data = list(set(data))
    unique_data.sort()
    return unique_data