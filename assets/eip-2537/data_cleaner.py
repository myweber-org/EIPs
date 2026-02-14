def clean_data(data):
    unique_data = list(set(data))
    sorted_data = sorted(unique_data)
    return sorted_data

def process_numbers(numbers):
    cleaned = clean_data(numbers)
    total = sum(cleaned)
    average = total / len(cleaned) if cleaned else 0
    return {
        'cleaned_data': cleaned,
        'total': total,
        'average': average
    }

if __name__ == "__main__":
    sample_data = [5, 2, 8, 2, 5, 9, 1, 8]
    result = process_numbers(sample_data)
    print(f"Cleaned data: {result['cleaned_data']}")
    print(f"Total: {result['total']}")
    print(f"Average: {result['average']}")