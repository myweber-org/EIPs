
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    if data.empty:
        return {"mean": np.nan, "median": np.nan, "std": np.nan}
    
    return {
        "mean": data[column].mean(),
        "median": data[column].median(),
        "std": data[column].std()
    }import csv
import re
from typing import List, Optional

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

def read_and_clean_csv(filepath: str) -> List[dict]:
    """Read CSV file and clean all string and numeric fields."""
    cleaned_data = []
    
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            cleaned_row = {}
            for key, value in row.items():
                if any(num_term in key.lower() for num_term in ['price', 'amount', 'quantity', 'total']):
                    cleaned_row[key] = clean_numeric(value)
                else:
                    cleaned_row[key] = clean_string(value)
            cleaned_data.append(cleaned_row)
    
    return cleaned_data

def write_cleaned_csv(data: List[dict], output_path: str) -> None:
    """Write cleaned data to a new CSV file."""
    if not data:
        return
    
    fieldnames = data[0].keys()
    
    with open(output_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def validate_data(data: List[dict]) -> List[dict]:
    """Remove rows with critical missing values."""
    validated = []
    
    for row in data:
        if all(row.get(key) is not None for key in ['id', 'name']):
            validated.append(row)
    
    return validated

def process_csv_pipeline(input_file: str, output_file: str) -> None:
    """Complete pipeline: read, clean, validate, and write CSV."""
    print(f"Processing {input_file}...")
    
    data = read_and_clean_csv(input_file)
    print(f"Cleaned {len(data)} rows")
    
    validated_data = validate_data(data)
    print(f"Validated {len(validated_data)} rows")
    
    write_cleaned_csv(validated_data, output_file)
    print(f"Results written to {output_file}")

if __name__ == "__main__":
    process_csv_pipeline("raw_data.csv", "cleaned_data.csv")