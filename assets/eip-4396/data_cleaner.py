def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    unique_list = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def clean_numeric_strings(data_list):
    """
    Convert string representations of numbers to integers where possible.
    Non-convertible items remain unchanged.
    """
    cleaned = []
    for item in data_list:
        if isinstance(item, str) and item.isdigit():
            cleaned.append(int(item))
        else:
            cleaned.append(item)
    return cleaned

def process_data(raw_data):
    """
    Main processing function: removes duplicates and cleans numeric strings.
    """
    deduplicated = remove_duplicates(raw_data)
    processed = clean_numeric_strings(deduplicated)
    return processed

if __name__ == "__main__":
    sample_data = ["1", "2", "2", "3", "apple", "banana", "3", "42", "apple"]
    result = process_data(sample_data)
    print(f"Original: {sample_data}")
    print(f"Processed: {result}")