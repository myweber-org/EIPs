
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_list = [3, 1, 2, 1, 4, 3, 5, 2]
    cleaned = remove_duplicates_preserve_order(sample_list)
    print(f"Original: {sample_list}")
    print(f"Cleaned: {cleaned}")