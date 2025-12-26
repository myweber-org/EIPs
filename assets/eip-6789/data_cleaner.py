
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: bool, whether to remove duplicate rows
        fill_missing: str, method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    missing_before = df.isnull().sum().sum()
    
    if fill_missing == 'drop':
        df = df.dropna()
        print(f"Dropped rows with missing values")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].median())
    elif fill_missing == 'mode':
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
    
    missing_after = df.isnull().sum().sum()
    print(f"Missing values: {missing_before} before, {missing_after} after cleaning")
    print(f"Dataset shape: {original_shape} -> {df.shape}")
    
    return df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the dataset meets basic requirements.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of column names that must be present
        min_rows: minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if len(df) < min_rows:
        print(f"Error: Dataset has only {len(df)} rows, minimum required is {min_rows}")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, None],
        'category': ['A', 'B', 'B', 'A', 'C', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nCleaning dataset...")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    if validate_data(cleaned_df, required_columns=['id', 'value'], min_rows=3):
        print("\nData validation passed")
    else:
        print("\nData validation failed")