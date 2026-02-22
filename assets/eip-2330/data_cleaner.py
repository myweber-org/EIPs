
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values. 
                           Options: 'mean', 'median', 'drop', 'fill_zero'.
    outlier_threshold (float): Number of standard deviations for outlier detection.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif missing_strategy == 'fill_zero':
        cleaned_df = cleaned_df.fillna(0)
    else:
        raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    
    # Handle outliers using z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if cleaned_df[col].std() > 0:  # Avoid division by zero
            z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
            outliers = z_scores > outlier_threshold
            
            if outliers.any():
                # Replace outliers with column median
                col_median = cleaned_df[col].median()
                cleaned_df.loc[outliers, col] = col_median
    
    # Reset index if rows were dropped
    if missing_strategy == 'drop':
        cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
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
    
    return True, "DataFrame is valid"

# Example usage function
def process_sample_data():
    """Demonstrate the data cleaning functionality."""
    # Create sample data with missing values and outliers
    np.random.seed(42)
    data = {
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100)
    }
    
    # Introduce missing values
    for col in data:
        mask = np.random.random(100) < 0.1
        data[col][mask] = np.nan
    
    # Introduce outliers
    data['A'][0] = 100  # Extreme outlier
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Clean the data
    cleaned_df = clean_dataframe(df, missing_strategy='median', outlier_threshold=2.5)
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Missing values after cleaning:")
    print(cleaned_df.isnull().sum())
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
    print(f"\nValidation result: {is_valid}")
    print(f"Validation message: {message}")
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = process_sample_data()
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_data.head())