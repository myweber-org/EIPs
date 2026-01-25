
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values. 
                           Options: 'mean', 'median', 'drop', 'fill_zero'.
    outlier_threshold (float): Number of standard deviations to identify outliers.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    df_clean = df.copy()
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif missing_strategy == 'median':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    elif missing_strategy == 'drop':
        df_clean = df_clean.dropna(subset=numeric_cols)
    elif missing_strategy == 'fill_zero':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
    
    # Handle outliers using z-score method
    for col in numeric_cols:
        z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
        outlier_mask = z_scores > outlier_threshold
        
        if outlier_mask.any():
            # Cap outliers at threshold * standard deviation
            upper_bound = df_clean[col].mean() + outlier_threshold * df_clean[col].std()
            lower_bound = df_clean[col].mean() - outlier_threshold * df_clean[col].std()
            
            df_clean.loc[outlier_mask, col] = np.where(
                df_clean.loc[outlier_mask, col] > upper_bound,
                upper_bound,
                np.where(
                    df_clean.loc[outlier_mask, col] < lower_bound,
                    lower_bound,
                    df_clean.loc[outlier_mask, col]
                )
            )
    
    return df_clean

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
    
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],  # Contains outlier (100) and missing value
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    df_cleaned = clean_dataframe(df, missing_strategy='mean', outlier_threshold=2)
    print("Cleaned DataFrame:")
    print(df_cleaned)
    print("\n")
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(df_cleaned, required_columns=['A', 'B', 'C'])
    print(f"Validation: {is_valid}")
    print(f"Message: {message}")