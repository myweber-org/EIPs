
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from specified column using IQR method.
    
    Args:
        df: pandas DataFrame
        column: column name to process
    
    Returns:
        Cleaned DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    return df[mask].reset_index(drop=True)

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from multiple columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to clean (default: all numeric columns)
    
    Returns:
        Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[95, 'value'] = 500
    df.loc[96, 'value'] = -200
    
    print("Original DataFrame shape:", df.shape)
    print("Original value statistics:")
    print(df['value'].describe())
    
    cleaned_df = clean_numeric_data(df, columns=['value'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned value statistics:")
    print(cleaned_df['value'].describe())
    
    is_valid, message = validate_dataframe(cleaned_df, ['id', 'value', 'category'])
    print(f"\nDataFrame validation: {is_valid} - {message}")
    
    return cleaned_df

if __name__ == "__main__":
    result_df = example_usage()