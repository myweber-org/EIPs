
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data(input_list):
    if not isinstance(input_list, list):
        raise TypeError("Input must be a list")
    return remove_duplicates_preserve_order(input_list)

if __name__ == "__main__":
    sample_data = [3, 1, 2, 3, 4, 2, 5, 1, 6]
    cleaned = clean_data(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")