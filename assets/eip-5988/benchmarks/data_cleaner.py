import csv
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def read_csv_file(file_path: str) -> List[Dict[str, Any]]:
    """Read a CSV file and return a list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        logger.info(f"Successfully read {len(data)} rows from {file_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        raise
    return data

def remove_empty_rows(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove rows where all values are empty strings or None."""
    cleaned_data = []
    for row in data:
        if any(value not in (None, "") for value in row.values()):
            cleaned_data.append(row)
    logger.info(f"Removed {len(data) - len(cleaned_data)} empty rows")
    return cleaned_data

def standardize_column_names(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Standardize column names to lowercase with underscores."""
    if not data:
        return data
    
    standardized_data = []
    for row in data:
        new_row = {}
        for key, value in row.items():
            new_key = key.strip().lower().replace(' ', '_')
            new_row[new_key] = value
        standardized_data.append(new_row)
    logger.info("Column names standardized")
    return standardized_data

def convert_numeric_values(data: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Convert specified columns to numeric values where possible."""
    if not data:
        return data
    
    if columns is None:
        columns = list(data[0].keys())
    
    converted_data = []
    for row in data:
        new_row = row.copy()
        for col in columns:
            if col in new_row:
                try:
                    value = new_row[col]
                    if value is None or value == "":
                        new_row[col] = None
                    else:
                        new_row[col] = float(value)
                except (ValueError, TypeError):
                    pass
        converted_data.append(new_row)
    logger.info(f"Attempted numeric conversion on columns: {columns}")
    return converted_data

def write_csv_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """Write data to a CSV file."""
    if not data:
        logger.warning("No data to write")
        return
    
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(data[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"Successfully wrote {len(data)} rows to {file_path}")
    except Exception as e:
        logger.error(f"Error writing CSV file {file_path}: {e}")
        raise

def clean_csv_data(input_file: str, output_file: str) -> None:
    """Complete data cleaning pipeline for CSV files."""
    logger.info(f"Starting data cleaning for {input_file}")
    
    data = read_csv_file(input_file)
    data = remove_empty_rows(data)
    data = standardize_column_names(data)
    data = convert_numeric_values(data)
    
    write_csv_file(data, output_file)
    logger.info(f"Data cleaning completed. Output saved to {output_file}")