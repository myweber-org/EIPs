import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str): Path to save cleaned CSV file
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    
    # Read CSV file
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Store original shape
    original_shape = df.shape
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values based on strategy
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    elif missing_strategy == 'median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    elif missing_strategy == 'drop':
        df = df.dropna(subset=numeric_cols)
    
    # Fill remaining non-numeric columns with mode
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    # Remove rows with all NaN values
    df = df.dropna(how='all')
    
    # Save cleaned data if output path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    
    # Print cleaning summary
    print(f"Original data shape: {original_shape}")
    print(f"Cleaned data shape: {df.shape}")
    print(f"Removed duplicates: {original_shape[0] - df.shape[0]}")
    print(f"Missing values handled using '{missing_strategy}' strategy")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            validation_results['warnings'].append(f'Column {col} contains infinite values')
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6],
        'value': [10.5, np.nan, 15.2, 15.2, np.nan, 20.1],
        'category': ['A', 'B', 'A', 'A', None, 'C'],
        'score': [85, 92, np.nan, 85, 78, 95]
    }
    
    # Create sample DataFrame
    df_sample = pd.DataFrame(sample_data)
    
    # Save sample data
    df_sample.to_csv('sample_data.csv', index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data(
        input_path='sample_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='mean'
    )
    
    # Validate cleaned data
    validation = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
    print(f"Validation passed: {validation['is_valid']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")