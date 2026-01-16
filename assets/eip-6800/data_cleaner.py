
import pandas as pd
import hashlib

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Columns to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def generate_hash(row):
    """
    Generate a hash for a row to identify duplicates.
    
    Args:
        row (pd.Series): Row from DataFrame
    
    Returns:
        str: MD5 hash of the row
    """
    row_str = str(row.values.tolist()).encode('utf-8')
    return hashlib.md5(row_str).hexdigest()

def clean_dataframe(df, columns_to_check=None):
    """
    Clean DataFrame by removing duplicates and adding hash column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns_to_check (list, optional): Columns to check for duplicates
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying original
    cleaned_df = df.copy()
    
    # Add hash column for duplicate detection
    cleaned_df['row_hash'] = cleaned_df.apply(generate_hash, axis=1)
    
    # Remove duplicates based on hash or specified columns
    if columns_to_check:
        cleaned_df = remove_duplicates(cleaned_df, subset=columns_to_check)
    else:
        cleaned_df = remove_duplicates(cleaned_df, subset=['row_hash'])
    
    # Remove the temporary hash column
    cleaned_df = cleaned_df.drop(columns=['row_hash'])
    
    # Reset index
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate DataFrame for common data quality issues.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
    
    Returns:
        dict: Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'null_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 1, 4, 2],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'David', 'Bob'],
        'age': [25, 30, 35, 25, 40, 30],
        'city': ['NYC', 'LA', 'Chicago', 'NYC', 'Miami', 'LA']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_dataframe(df))
    
    cleaned_df = clean_dataframe(df, columns_to_check=['id', 'name'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Validation Results:")
    print(validate_dataframe(cleaned_df))