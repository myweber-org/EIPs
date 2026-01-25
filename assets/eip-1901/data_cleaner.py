
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = 'mean',
    columns_to_drop: Optional[list] = None
) -> pd.DataFrame:
    """
    Load and clean CSV data by handling missing values and optional column removal.
    
    Parameters:
    input_path: Path to input CSV file
    output_path: Path where cleaned CSV will be saved
    missing_strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns_to_drop: List of column names to remove from dataset
    
    Returns:
    Cleaned DataFrame
    """
    
    df = pd.read_csv(input_path)
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
        print(f"Dropped columns: {columns_to_drop}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif missing_strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif missing_strategy == 'mode':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    elif missing_strategy == 'drop':
        df = df.dropna(subset=numeric_cols)
    
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    df.to_csv(output_path, index=False)
    
    final_shape = df.shape
    print(f"Cleaned data shape: {final_shape}")
    print(f"Rows removed: {original_shape[0] - final_shape[0]}")
    print(f"Columns removed: {original_shape[1] - final_shape[1]}")
    print(f"Cleaned data saved to: {output_path}")
    
    return df

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate DataFrame for common data quality issues.
    
    Returns dictionary with validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(exclude=[np.number]).columns)
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        validation_results['numeric_stats'] = {
            col: {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            }
            for col in numeric_cols[:5]
        }
    
    return validation_results

if __name__ == "__main__":
    sample_df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, np.nan, np.nan, 8, 10],
        'C': ['X', 'Y', 'Z', np.nan, 'W'],
        'D': [100, 200, 300, 400, 500]
    })
    
    sample_df.to_csv('sample_data.csv', index=False)
    
    cleaned = clean_csv_data(
        input_path='sample_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='mean',
        columns_to_drop=['D']
    )
    
    validation = validate_dataframe(cleaned)
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (bool): Whether to fill missing values. Default is True.
        fill_value: Value to use for filling missing data. Default is 0.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False
    
    return True

def process_data_file(file_path, output_path=None):
    """
    Load, clean, and optionally save a dataset from a CSV file.
    
    Args:
        file_path (str): Path to the input CSV file.
        output_path (str, optional): Path to save the cleaned data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        
        if not validate_dataframe(df):
            raise ValueError("DataFrame validation failed")
        
        cleaned_df = clean_dataset(df)
        
        if output_path:
            cleaned_df.to_csv(output_path, index=False)
        
        return cleaned_df
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None