import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    if subset is None:
        return df.drop_duplicates()
    else:
        return df.drop_duplicates(subset=subset)

def convert_column_types(df, column_type_map):
    """
    Convert specified columns to given data types.
    
    Args:
        df: pandas DataFrame
        column_type_map: dict mapping column names to target types
    
    Returns:
        DataFrame with converted column types
    """
    df_converted = df.copy()
    for column, dtype in column_type_map.items():
        if column in df_converted.columns:
            try:
                df_converted[column] = df_converted[column].astype(dtype)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert column '{column}' to {dtype}")
    
    return df_converted

def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill values
        fill_value: value to use when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        if fill_value is not None:
            return df.fillna(fill_value)
        else:
            return df.fillna(0)
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")

def clean_dataframe(df, config):
    """
    Apply multiple cleaning operations based on configuration.
    
    Args:
        df: pandas DataFrame to clean
        config: dict with cleaning configuration
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if config.get('remove_duplicates'):
        subset = config.get('duplicate_subset')
        cleaned_df = remove_duplicates(cleaned_df, subset)
    
    if config.get('convert_types'):
        type_map = config.get('type_mapping', {})
        cleaned_df = convert_column_types(cleaned_df, type_map)
    
    if config.get('handle_missing'):
        strategy = config.get('missing_strategy', 'drop')
        fill_value = config.get('fill_value')
        cleaned_df = handle_missing_values(cleaned_df, strategy, fill_value)
    
    return cleaned_df

def validate_dataframe(df, validation_rules):
    """
    Validate DataFrame against specified rules.
    
    Args:
        df: pandas DataFrame to validate
        validation_rules: dict with validation rules
    
    Returns:
        dict with validation results
    """
    results = {}
    
    for column, rules in validation_rules.items():
        if column in df.columns:
            column_results = {}
            
            if 'min' in rules:
                column_results['min_valid'] = df[column].min() >= rules['min']
            
            if 'max' in rules:
                column_results['max_valid'] = df[column].max() <= rules['max']
            
            if 'unique' in rules:
                column_results['unique_valid'] = df[column].nunique() == len(df[column])
            
            results[column] = column_results
    
    return results
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (bool): Whether to fill missing values
        fill_value: Value to use for filling missing data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.fillna(fill_value)
        missing_after = cleaned_df.isnull().sum().sum()
        print(f"Filled {missing_before - missing_after} missing values")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10, None, 20, 30, None],
        'category': ['A', 'B', 'B', 'C', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df)
    print(cleaned)