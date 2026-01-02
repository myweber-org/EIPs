
import csv
import re
from typing import List, Optional

def clean_csv_data(input_file: str, output_file: str, columns_to_clean: Optional[List[str]] = None) -> None:
    """
    Clean a CSV file by removing extra whitespace and standardizing text.
    
    Args:
        input_file: Path to the input CSV file.
        output_file: Path to save the cleaned CSV file.
        columns_to_clean: List of column names to apply cleaning. If None, clean all columns.
    """
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        if fieldnames is None:
            raise ValueError("CSV file has no headers")
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        columns_to_process = columns_to_clean if columns_to_clean else fieldnames
        
        for row in reader:
            cleaned_row = {}
            for field in fieldnames:
                value = row.get(field, '')
                if field in columns_to_process and isinstance(value, str):
                    # Remove extra whitespace
                    value = re.sub(r'\s+', ' ', value.strip())
                    # Standardize capitalization for certain patterns
                    value = value.lower() if value.isupper() else value
                cleaned_row[field] = value
            writer.writerow(cleaned_row)

def validate_email_in_column(csv_file: str, email_column: str) -> List[dict]:
    """
    Validate email addresses in a specific column of a CSV file.
    
    Args:
        csv_file: Path to the CSV file.
        email_column: Name of the column containing email addresses.
    
    Returns:
        List of dictionaries with validation results.
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    results = []
    
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for i, row in enumerate(reader, start=2):  # start=2 for line numbers (header is line 1)
            email = row.get(email_column, '')
            is_valid = bool(re.match(email_pattern, email))
            
            results.append({
                'line': i,
                'email': email,
                'is_valid': is_valid,
                'original_row': row
            })
    
    return results

def remove_duplicates(csv_file: str, unique_columns: List[str], output_file: str) -> None:
    """
    Remove duplicate rows based on specified columns.
    
    Args:
        csv_file: Path to the input CSV file.
        unique_columns: List of column names to identify duplicates.
        output_file: Path to save the deduplicated CSV file.
    """
    seen = set()
    unique_rows = []
    
    with open(csv_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        if fieldnames is None:
            raise ValueError("CSV file has no headers")
        
        for row in reader:
            # Create a tuple of values from the specified unique columns
            key_tuple = tuple(str(row.get(col, '')).strip().lower() for col in unique_columns)
            
            if key_tuple not in seen:
                seen.add(key_tuple)
                unique_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)

if __name__ == "__main__":
    # Example usage
    clean_csv_data("raw_data.csv", "cleaned_data.csv")
    
    validation_results = validate_email_in_column("cleaned_data.csv", "email")
    invalid_emails = [r for r in validation_results if not r['is_valid']]
    print(f"Found {len(invalid_emails)} invalid email addresses")
    
    remove_duplicates("cleaned_data.csv", ["email", "name"], "deduplicated_data.csv")