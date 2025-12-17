
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
    Clean CSV data by handling missing values and standardizing columns.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
        missing_strategy: Strategy for handling missing values ('drop', 'fill', 'interpolate')
        fill_value: Value to use when missing_strategy is 'fill'
    
    Returns:
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(input_path)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Handle missing values
        if missing_strategy == 'drop':
            df = df.dropna()
        elif missing_strategy == 'fill':
            if fill_value is not None:
                df = df.fillna(fill_value)
            else:
                df = df.fillna(df.mean(numeric_only=True))
        elif missing_strategy == 'interpolate':
            df = df.interpolate(method='linear', limit_direction='forward')
        
        # Convert column names to lowercase with underscores
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Remove leading/trailing whitespace from string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].str.strip()
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        
        print(f"Data cleaning completed. Cleaned data saved to {output_path}")
        print(f"Original shape: {df.shape}, Cleaned shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
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
        Boolean indicating if data passes validation
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    # Check for infinite values
    if np.any(np.isinf(df.select_dtypes(include=[np.number]))):
        print("Validation warning: DataFrame contains infinite values")
    
    # Check for negative values in columns that shouldn't have them
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if 'price' in col or 'cost' in col or 'amount' in col:
            if (df[col] < 0).any():
                print(f"Validation warning: Column '{col}' contains negative values")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_df = clean_csv_data(
        input_path='raw_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='fill',
        fill_value=0
    )
    
    if validate_dataframe(sample_df):
        print("Data validation passed")
    else:
        print("Data validation failed")