
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    original_shape = df.shape
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    removed_count = original_shape[0] - cleaned_df.shape[0]
    print(f"Removed {removed_count} outliers from dataset")
    return cleaned_dfimport pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str, optional): Path for cleaned output CSV
    strategy (str): Method for handling missing values ('mean', 'median', 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    df = df.drop_duplicates()
    print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    missing_before = df.isnull().sum().sum()
    
    if strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'drop':
        df = df.dropna()
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
    
    missing_after = df.isnull().sum().sum()
    print(f"Missing values handled: {missing_before} -> {missing_after}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    print(f"Final data shape: {df.shape}")
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary': {}
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing columns: {missing_cols}")
    
    validation_results['summary']['rows'] = len(df)
    validation_results['summary']['columns'] = len(df.columns)
    validation_results['summary']['memory_usage'] = df.memory_usage(deep=True).sum()
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, np.nan, np.nan, 8, 10],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_data.csv', strategy='mean')
    
    validation = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
    print(f"Validation results: {validation}")