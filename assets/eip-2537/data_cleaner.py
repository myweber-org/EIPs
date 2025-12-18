import csv
import re
from typing import List, Dict, Optional

def clean_string(value: str) -> str:
    """Remove extra whitespace and normalize string."""
    if not isinstance(value, str):
        return str(value)
    cleaned = re.sub(r'\s+', ' ', value.strip())
    return cleaned

def clean_numeric(value: str) -> Optional[float]:
    """Convert string to float, handling common issues."""
    if not value:
        return None
    cleaned = value.replace(',', '').replace('$', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return None

def validate_email(email: str) -> bool:
    """Basic email validation."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))

def read_csv_file(filepath: str) -> List[Dict]:
    """Read CSV file and return list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return data

def clean_csv_data(data: List[Dict]) -> List[Dict]:
    """Apply cleaning functions to CSV data."""
    cleaned_data = []
    for row in data:
        cleaned_row = {}
        for key, value in row.items():
            if key.endswith('_email'):
                cleaned_row[key] = value if validate_email(value) else None
            elif any(num_key in key for num_key in ['price', 'amount', 'quantity']):
                cleaned_row[key] = clean_numeric(value)
            else:
                cleaned_row[key] = clean_string(value)
        cleaned_data.append(cleaned_row)
    return cleaned_data

def write_cleaned_csv(data: List[Dict], output_path: str) -> None:
    """Write cleaned data to new CSV file."""
    if not data:
        return
    fieldnames = data[0].keys()
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Cleaned data written to: {output_path}")
    except Exception as e:
        print(f"Error writing CSV: {e}")

def process_csv_pipeline(input_file: str, output_file: str) -> None:
    """Complete pipeline for CSV data cleaning."""
    raw_data = read_csv_file(input_file)
    if raw_data:
        cleaned_data = clean_csv_data(raw_data)
        write_cleaned_csv(cleaned_data, output_file)