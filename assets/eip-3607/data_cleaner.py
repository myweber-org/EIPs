
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Load a CSV file, clean missing values, and save cleaned data.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values.")
            
            # Handle missing values based on strategy
            if missing_strategy == 'mean':
                df = df.fillna(df.mean(numeric_only=True))
            elif missing_strategy == 'median':
                df = df.fillna(df.median(numeric_only=True))
            elif missing_strategy == 'mode':
                df = df.fillna(df.mode().iloc[0])
            elif missing_strategy == 'drop':
                df = df.dropna()
            else:
                df = df.fillna(0)
                
            print(f"Missing values handled using '{missing_strategy}' strategy.")
        
        # Remove duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            print(f"Removed {duplicates} duplicate rows.")
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        print(f"Final data shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None

def validate_data(df, required_columns=None):
    """
    Validate data structure and content.
    """
    if df is None or df.empty:
        print("Error: Data is empty or None.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    # Check for negative values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] < 0).any():
            print(f"Warning: Column '{col}' contains negative values.")
    
    return True

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_csv, output_csv, missing_strategy='median')
    
    if cleaned_df is not None:
        is_valid = validate_data(cleaned_df)
        if is_valid:
            print("Data cleaning completed successfully.")
        else:
            print("Data validation failed.")
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = 'drop',
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and standardizing column names.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
        missing_strategy: Strategy for handling missing values ('drop', 'fill', 'interpolate')
        fill_value: Value to fill missing data with when using 'fill' strategy
    
    Returns:
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(input_path)
        
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'drop':
            df_cleaned = df.dropna(subset=numeric_cols)
        elif missing_strategy == 'fill':
            if fill_value is not None:
                df_cleaned = df.fillna({col: fill_value for col in numeric_cols})
            else:
                df_cleaned = df.fillna(df[numeric_cols].mean())
        elif missing_strategy == 'interpolate':
            df_cleaned = df.copy()
            for col in numeric_cols:
                df_cleaned[col] = df_cleaned[col].interpolate(method='linear')
        else:
            raise ValueError(f"Unknown missing strategy: {missing_strategy}")
        
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        print(f"Original rows: {len(df)}, Cleaned rows: {len(df_cleaned)}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        raise
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty")
        raise
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        raise

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame for common data quality issues.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if validation passes, False otherwise
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if df.isnull().all().any():
        print("Validation failed: Some columns contain only null values")
        return False
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].abs().max() > 1e10:
            print(f"Validation warning: Column {col} contains extremely large values")
    
    return True

if __name__ == "__main__":
    sample_df = pd.DataFrame({
        'temperature': [22.5, np.nan, 24.0, 25.5, np.nan],
        'humidity': [45, 50, np.nan, 55, 60],
        'pressure': [1013, 1012, 1011, np.nan, 1010]
    })
    
    sample_df.to_csv('sample_data.csv', index=False)
    
    cleaned = clean_csv_data(
        input_path='sample_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='fill',
        fill_value=0
    )
    
    if validate_dataframe(cleaned):
        print("Data validation passed")
    else:
        print("Data validation failed")
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
        raise ValueError("Unsupported strategy. Use 'mean', 'median', 'mode', or 'drop'")

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if outlier_method == 'iqr':
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            elif outlier_method == 'zscore':
                cleaned_df = remove_outliers_zscore(cleaned_df, col)
            
            if normalize_method == 'minmax':
                cleaned_df = normalize_minmax(cleaned_df, col)
            elif normalize_method == 'zscore':
                cleaned_df = normalize_zscore(cleaned_df, col)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    return cleaned_dfimport csv
import re
from typing import List, Dict, Any, Optional

def clean_csv_data(input_file: str, output_file: str, columns_to_clean: Optional[List[str]] = None) -> None:
    """
    Clean a CSV file by removing extra whitespace and standardizing text.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output cleaned CSV file
        columns_to_clean: List of column names to clean. If None, clean all columns.
    """
    
    def clean_text(text: str) -> str:
        """Remove extra whitespace and standardize text."""
        if not isinstance(text, str):
            return text
            
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Standardize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text
    
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            
            if columns_to_clean is None:
                columns_to_clean = fieldnames
            
            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in reader:
                    cleaned_row = {}
                    for field in fieldnames:
                        value = row.get(field, '')
                        if field in columns_to_clean:
                            cleaned_row[field] = clean_text(value)
                        else:
                            cleaned_row[field] = value
                    writer.writerow(cleaned_row)
                    
        print(f"Successfully cleaned data. Output saved to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def validate_email(email: str) -> bool:
    """
    Validate email format using regex.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email is valid, False otherwise
    """
    if not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def remove_duplicates(data: List[Dict[str, Any]], key_field: str) -> List[Dict[str, Any]]:
    """
    Remove duplicate rows based on a key field.
    
    Args:
        data: List of dictionaries representing rows
        key_field: Field name to use for duplicate detection
        
    Returns:
        List with duplicates removed
    """
    seen = set()
    unique_data = []
    
    for row in data:
        key_value = row.get(key_field)
        if key_value not in seen:
            seen.add(key_value)
            unique_data.append(row)
    
    return unique_data

if __name__ == "__main__":
    # Example usage
    sample_data = [
        {"id": 1, "name": "John Doe  ", "email": "john@example.com"},
        {"id": 2, "name": "Jane Smith", "email": "jane@example.com"},
        {"id": 1, "name": "John Doe", "email": "john@example.com"}  # Duplicate
    ]
    
    # Test duplicate removal
    unique_data = remove_duplicates(sample_data, "id")
    print(f"Original: {len(sample_data)} rows, Unique: {len(unique_data)} rows")
    
    # Test email validation
    test_emails = ["test@example.com", "invalid-email", "another@test.co.uk"]
    for email in test_emails:
        is_valid = validate_email(email)
        print(f"{email}: {'Valid' if is_valid else 'Invalid'}")