import csv
import re
from typing import List, Dict, Any, Optional

def clean_string(value: str) -> str:
    """Remove extra whitespace and normalize string."""
    if not isinstance(value, str):
        return str(value) if value is not None else ""
    return re.sub(r'\s+', ' ', value.strip())

def clean_numeric(value: Any) -> Optional[float]:
    """Convert value to float, handling common issues."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(',', '').strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None

def clean_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Apply cleaning functions to all values in a row."""
    cleaned = {}
    for key, value in row.items():
        if isinstance(value, str):
            cleaned[key] = clean_string(value)
        elif isinstance(value, (int, float)):
            cleaned[key] = clean_numeric(value)
        else:
            cleaned[key] = value
    return cleaned

def read_and_clean_csv(filepath: str) -> List[Dict[str, Any]]:
    """Read CSV file and clean all rows."""
    cleaned_data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                cleaned_data.append(clean_csv_row(row))
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return cleaned_data

def write_cleaned_csv(data: List[Dict[str, Any]], output_path: str) -> bool:
    """Write cleaned data to a new CSV file."""
    if not data:
        return False
    try:
        fieldnames = data[0].keys()
        with open(output_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing CSV: {e}")
        return False

def validate_email(email: str) -> bool:
    """Simple email validation."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email)) if email else False

def clean_phone_number(phone: str) -> str:
    """Standardize phone number format."""
    if not phone:
        return ""
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return digits