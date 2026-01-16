
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

def clean_numeric_strings(string_list):
    """
    Clean a list of numeric strings by converting to integers
    and removing any non-numeric entries.
    """
    cleaned = []
    for s in string_list:
        try:
            cleaned.append(int(s))
        except ValueError:
            continue
    return cleaned

if __name__ == "__main__":
    sample_data = ["1", "2", "2", "3", "four", "5", "5"]
    unique_data = remove_duplicates(sample_data)
    numeric_data = clean_numeric_strings(unique_data)
    print(f"Original: {sample_data}")
    print(f"Unique: {unique_data}")
    print(f"Numeric: {numeric_data}")