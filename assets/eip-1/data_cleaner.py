import re
from typing import List, Set

def clean_text(text: str) -> str:
    """Standardize text by converting to lowercase and removing extra whitespace."""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_duplicates(items: List[str]) -> List[str]:
    """Remove duplicate items while preserving order."""
    seen: Set[str] = set()
    unique_items = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)
    return unique_items

def process_data(data: List[str]) -> List[str]:
    """Clean and deduplicate a list of text strings."""
    cleaned = [clean_text(item) for item in data]
    return remove_duplicates(cleaned)

if __name__ == "__main__":
    sample_data = ["Hello World", "hello world", "  Test  ", "test", "Python"]
    result = process_data(sample_data)
    print(f"Original: {sample_data}")
    print(f"Processed: {result}")