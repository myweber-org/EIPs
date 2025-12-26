
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fill_missing: Strategy for filling missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    original_shape = df.shape
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = original_shape[0] - cleaned_df.shape[0]
        print(f"Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values
    missing_before = cleaned_df.isnull().sum().sum()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
    
    missing_after = cleaned_df.isnull().sum().sum()
    print(f"Handled {missing_before - missing_after} missing values")
    
    # Report cleaning statistics
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    
    return cleaned_df

def validate_data(df, required_columns=None, numeric_ranges=None):
    """
    Validate data quality after cleaning.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of columns that must be present
        numeric_ranges: Dict of column names and (min, max) acceptable ranges
    
    Returns:
        Boolean indicating if data passed validation
    """
    validation_passed = True
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            validation_passed = False
    
    # Check numeric ranges
    if numeric_ranges:
        for col, (min_val, max_val) in numeric_ranges.items():
            if col in df.columns and df[col].dtype in [np.int64, np.float64]:
                out_of_range = df[(df[col] < min_val) | (df[col] > max_val)].shape[0]
                if out_of_range > 0:
                    print(f"Column '{col}' has {out_of_range} values outside range [{min_val}, {max_val}]")
                    validation_passed = False
    
    # Check for remaining missing values
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Data still contains {remaining_missing} missing values")
        validation_passed = False
    
    return validation_passed

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        'id': [1, 2, 3, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Charlie', None, 'Eve'],
        'age': [25, 30, None, 35, 28, 22],
        'score': [85.5, 92.0, 78.5, 78.5, None, 95.0]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    
    print("\n" + "="*50 + "\n")
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    print("\n" + "="*50 + "\n")
    print("Data Validation:")
    validation_rules = {
        'required_columns': ['id', 'name', 'age', 'score'],
        'numeric_ranges': {'age': (18, 100), 'score': (0, 100)}
    }
    
    is_valid = validate_data(
        cleaned_df,
        required_columns=validation_rules['required_columns'],
        numeric_ranges=validation_rules['numeric_ranges']
    )
    
    print(f"Data validation passed: {is_valid}")