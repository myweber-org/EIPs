import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
        keep: {'first', 'last', False} which duplicates to keep
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    original_shape = df.shape
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    removed_count = original_shape[0] - cleaned_df.shape[0]
    
    print(f"Removed {removed_count} duplicate rows")
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    
    return cleaned_df

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: {'mean', 'median', 'mode', 'drop', 'fill'}
        columns: list of columns to apply cleaning to
    
    Returns:
        DataFrame with missing values handled
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_cleaned = df.copy()
    
    for col in columns:
        if col in df.columns:
            if strategy == 'drop':
                df_cleaned = df_cleaned.dropna(subset=[col])
            elif strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            elif strategy == 'mode':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
            elif strategy == 'fill':
                df_cleaned[col] = df_cleaned[col].fillna(0)
    
    return df_cleaned

def validate_data(df, rules):
    """
    Validate DataFrame against specified rules.
    
    Args:
        df: pandas DataFrame
        rules: dictionary of validation rules
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'passed': True,
        'errors': [],
        'warnings': []
    }
    
    for column, rule in rules.items():
        if column in df.columns:
            if 'min' in rule and df[column].min() < rule['min']:
                results['passed'] = False
                results['errors'].append(f"{column}: value below minimum {rule['min']}")
            
            if 'max' in rule and df[column].max() > rule['max']:
                results['passed'] = False
                results['errors'].append(f"{column}: value above maximum {rule['max']}")
            
            if 'unique' in rule and rule['unique']:
                if df[column].nunique() != len(df):
                    results['warnings'].append(f"{column}: contains duplicate values")
    
    return results

def main():
    # Example usage
    data = {
        'id': [1, 2, 2, 3, 4, 4, 5],
        'value': [10, 20, 20, 30, None, 50, 60],
        'category': ['A', 'B', 'B', 'C', 'D', 'D', 'E']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Remove duplicates
    df_clean = remove_duplicates(df, subset=['id', 'category'])
    print()
    
    # Clean missing values
    df_clean = clean_missing_values(df_clean, strategy='mean')
    print("DataFrame after cleaning:")
    print(df_clean)
    print()
    
    # Validate data
    validation_rules = {
        'id': {'min': 1, 'max': 100},
        'value': {'min': 0, 'max': 100}
    }
    
    validation_results = validate_data(df_clean, validation_rules)
    print("Validation Results:")
    print(f"Passed: {validation_results['passed']}")
    print(f"Errors: {validation_results['errors']}")
    print(f"Warnings: {validation_results['warnings']}")

if __name__ == "__main__":
    main()