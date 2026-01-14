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
    return digitsimport numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def process_dataframe(df, numeric_columns):
    """
    Process DataFrame by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: Processed DataFrame with outliers removed
    """
    processed_df = df.copy()
    
    for column in numeric_columns:
        if column in processed_df.columns and pd.api.types.is_numeric_dtype(processed_df[column]):
            original_count = len(processed_df)
            processed_df = remove_outliers_iqr(processed_df, column)
            removed_count = original_count - len(processed_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return processed_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[::100, 'A'] = 500
    
    print("Original DataFrame shape:", sample_df.shape)
    print("\nOriginal summary statistics for column 'A':")
    print(calculate_summary_statistics(sample_df, 'A'))
    
    processed_df = process_dataframe(sample_df, ['A', 'B', 'C'])
    
    print("\nProcessed DataFrame shape:", processed_df.shape)
    print("\nProcessed summary statistics for column 'A':")
    print(calculate_summary_statistics(processed_df, 'A'))
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing NaN values and converting to appropriate types.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to clean. If None, clean all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            cleaned_df = cleaned_df.dropna(subset=[col])
    
    return cleaned_df

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    data = {
        'id': range(1, 11),
        'value': [10, 12, 13, 14, 15, 16, 17, 18, 19, 100]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    print("DataFrame after outlier removal:")
    print(cleaned_df)
    print()
    
    stats = calculate_basic_stats(df, 'value')
    print("Statistics for 'value' column:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    example_usage()