
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