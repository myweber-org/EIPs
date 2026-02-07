
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5, 1, 6]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    strategy (str): Strategy for missing value imputation ('mean', 'median', 'mode', 'drop').
    outlier_threshold (float): Number of standard deviations to identify outliers.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Handle missing values
    if strategy == 'drop':
        df_clean = df_clean.dropna()
    elif strategy == 'mean':
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    elif strategy == 'median':
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    elif strategy == 'mode':
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    
    # Handle outliers for numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_mean = df_clean[col].mean()
        col_std = df_clean[col].std()
        
        if col_std > 0:  # Avoid division by zero
            z_scores = np.abs((df_clean[col] - col_mean) / col_std)
            df_clean = df_clean[z_scores < outlier_threshold]
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],  # Contains outlier (100)
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Clean the data
    df_clean = clean_dataset(df, strategy='median', outlier_threshold=2)
    print("Cleaned DataFrame:")
    print(df_clean)
    print()
    
    # Validate the cleaned data
    is_valid, message = validate_data(df_clean, required_columns=['A', 'B', 'C'], min_rows=2)
    print(f"Validation: {is_valid}")
    print(f"Message: {message}")