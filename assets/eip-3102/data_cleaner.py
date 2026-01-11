
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
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
    Clean the input data by removing duplicates.
    Handles both list and tuple inputs.
    """
    if isinstance(data, (list, tuple)):
        return remove_duplicates(list(data))
    else:
        raise TypeError("Input must be a list or tuple")

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5, 1, 6]
    cleaned = clean_data(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")