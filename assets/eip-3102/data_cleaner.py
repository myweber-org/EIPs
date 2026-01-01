import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df: Input pandas DataFrame
        column_mapping: Dictionary to rename columns
        drop_duplicates: Whether to drop duplicate rows
        fill_missing: Strategy to fill missing values ('mean', 'median', 'mode', or value)
    
    Returns:
        Cleaned pandas DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Rename columns if mapping provided
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Drop duplicate rows
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    # Handle missing values
    for column in cleaned_df.columns:
        if cleaned_df[column].isnull().any():
            missing_count = cleaned_df[column].isnull().sum()
            print(f"Column '{column}' has {missing_count} missing values")
            
            if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                fill_value = cleaned_df[column].mean()
            elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                fill_value = cleaned_df[column].median()
            elif fill_missing == 'mode':
                fill_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else np.nan
            else:
                fill_value = fill_missing
            
            cleaned_df[column] = cleaned_df[column].fillna(fill_value)
            print(f"Filled missing values in '{column}' with {fill_value}")
    
    # Convert object columns to appropriate types
    for column in cleaned_df.select_dtypes(include=['object']).columns:
        try:
            cleaned_df[column] = pd.to_datetime(cleaned_df[column])
            print(f"Converted column '{column}' to datetime")
        except (ValueError, TypeError):
            try:
                cleaned_df[column] = pd.to_numeric(cleaned_df[column])
                print(f"Converted column '{column}' to numeric")
            except (ValueError, TypeError):
                pass
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, numeric_ranges=None):
    """
    Validate data quality after cleaning.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        numeric_ranges: Dict of column: (min, max) for numeric validation
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'has_data': len(df) > 0,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum()
    }
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing_columns
        validation_results['all_required_columns'] = len(missing_columns) == 0
    
    # Validate numeric ranges
    if numeric_ranges:
        range_violations = {}
        for column, (min_val, max_val) in numeric_ranges.items():
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                below_min = (df[column] < min_val).sum()
                above_max = (df[column] > max_val).sum()
                if below_min > 0 or above_max > 0:
                    range_violations[column] = {
                        'below_min': below_min,
                        'above_max': above_max
                    }
        validation_results['range_violations'] = range_violations
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-02', 'invalid', '2023-01-04', '2023-01-05']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='median')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    validation = validate_data(
        cleaned_df,
        required_columns=['id', 'value', 'category'],
        numeric_ranges={'value': (0, 100)}
    )
    
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")