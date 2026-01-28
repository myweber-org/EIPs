
import csv
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def read_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Read a CSV file and return its contents as a list of dictionaries.
    """
    data = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
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

def clean_empty_rows(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove rows where all values are empty strings.
    """
    cleaned_data = []
    for row in data:
        if any(value.strip() for value in row.values()):
            cleaned_data.append(row)
    logger.info(f"Removed {len(data) - len(cleaned_data)} empty rows")
    return cleaned_data

def normalize_column_names(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize column names to lowercase and replace spaces with underscores.
    """
    if not data:
        return data
    normalized_data = []
    for row in data:
        new_row = {}
        for key, value in row.items():
            new_key = key.strip().lower().replace(' ', '_')
            new_row[new_key] = value
        normalized_data.append(new_row)
    logger.info("Column names normalized")
    return normalized_data

def write_csv(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Write data to a CSV file.
    """
    if not data:
        logger.warning("No data to write")
        return
    try:
        fieldnames = data[0].keys()
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"Successfully wrote {len(data)} rows to {file_path}")
    except Exception as e:
        logger.error(f"Error writing CSV file {file_path}: {e}")
        raise

def clean_csv(input_path: str, output_path: str) -> None:
    """
    Main function to read, clean, and write a CSV file.
    """
    data = read_csv(input_path)
    data = clean_empty_rows(data)
    data = normalize_column_names(data)
    write_csv(data, output_path)