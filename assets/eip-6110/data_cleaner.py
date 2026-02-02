
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
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
    
    return True, "Dataset is valid"import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(df, column_name):
    """
    Clean a numeric column by removing non-numeric characters.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the column to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df

def validate_dataframe(df, required_columns):
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if all required columns are present.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return Trueimport pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_method='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    filepath (str): Path to the CSV file
    fill_method (str): Method for filling missing values ('mean', 'median', 'mode', 'zero')
    drop_threshold (float): Drop columns with missing ratio above this threshold (0.0-1.0)
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {filepath}")
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    # Calculate missing ratio per column
    missing_ratio = df.isnull().sum() / len(df)
    
    # Drop columns with high missing ratio
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    if len(columns_to_drop) > 0:
        df = df.drop(columns=columns_to_drop)
        print(f"Dropped columns with >{drop_threshold*100}% missing values: {list(columns_to_drop)}")
    
    # Handle remaining missing values
    for column in df.columns:
        if df[column].isnull().any():
            if fill_method == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                fill_value = df[column].mean()
            elif fill_method == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                fill_value = df[column].median()
            elif fill_method == 'mode':
                fill_value = df[column].mode()[0] if not df[column].mode().empty else np.nan
            elif fill_method == 'zero':
                fill_value = 0
            else:
                fill_value = df[column].ffill().bfill().iloc[0] if not df[column].empty else np.nan
            
            df[column] = df[column].fillna(fill_value)
    
    # Remove duplicate rows
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates_count} duplicate rows")
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    final_shape = df.shape
    print(f"Final data shape: {final_shape}")
    print(f"Rows removed: {original_shape[0] - final_shape[0]}")
    print(f"Columns removed: {original_shape[1] - final_shape[1]}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary': {}
    }
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append('DataFrame is empty')
        return validation_results
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing required columns: {missing_columns}')
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        if inf_count > 0:
            validation_results['issues'].append(f'Found {inf_count} infinite values in numeric columns')
    
    # Summary statistics
    validation_results['summary'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.to_dict()
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 1, 1, 1, 1],
        'D': [10, 20, 30, 40, 50]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_method='median', drop_threshold=0.3)
    validation = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'D'])
    
    print("\nValidation Results:")
    print(f"Is valid: {validation['is_valid']}")
    print(f"Issues: {validation['issues']}")
    print(f"Summary: {validation['summary']}")
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result