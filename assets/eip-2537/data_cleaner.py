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
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def handle_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'drop':
        return df.dropna()
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'drop'")

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            if outlier_method == 'iqr':
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            elif outlier_method == 'zscore':
                cleaned_df = remove_outliers_zscore(cleaned_df, column)
            
            if normalize_method == 'minmax':
                cleaned_df = normalize_minmax(cleaned_df, column)
            elif normalize_method == 'zscore':
                cleaned_df = normalize_zscore(cleaned_df, column)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'feature1': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature3': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    numeric_cols = ['feature1', 'feature2', 'feature3']
    cleaned = clean_dataset(df, numeric_cols, outlier_method='iqr', normalize_method='minmax')
    
    print("\nCleaned DataFrame:")
    print(cleaned)
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    if strategy == 'mean':
        return data.fillna(data.mean())
    elif strategy == 'median':
        return data.fillna(data.median())
    elif strategy == 'mode':
        return data.fillna(data.mode().iloc[0])
    elif strategy == 'drop':
        return data.dropna()
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'drop'")

def clean_dataset(data, numeric_columns, outlier_columns=None, normalize_columns=None, standardize_columns=None, missing_strategy='mean'):
    cleaned_data = data.copy()
    
    if outlier_columns:
        for col in outlier_columns:
            if col in numeric_columns:
                cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    if normalize_columns:
        for col in normalize_columns:
            if col in numeric_columns:
                cleaned_data[col + '_normalized'] = normalize_minmax(cleaned_data, col)
    
    if standardize_columns:
        for col in standardize_columns:
            if col in numeric_columns:
                cleaned_data[col + '_standardized'] = standardize_zscore(cleaned_data, col)
    
    return cleaned_data

def validate_data(data, required_columns, numeric_ranges=None):
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if numeric_ranges:
        for col, (min_val, max_val) in numeric_ranges.items():
            if col in data.columns:
                invalid_values = data[(data[col] < min_val) | (data[col] > max_val)]
                if not invalid_values.empty:
                    print(f"Warning: Column '{col}' contains values outside range [{min_val}, {max_val}]")
    
    return True