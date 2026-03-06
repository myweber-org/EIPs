
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_dataframe': isinstance(df, pd.DataFrame),
        'has_data': not df.empty if isinstance(df, pd.DataFrame) else False,
        'columns': list(df.columns) if isinstance(df, pd.DataFrame) else [],
        'shape': df.shape if isinstance(df, pd.DataFrame) else (0, 0),
        'missing_columns': []
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing
        validation_results['all_required_present'] = len(missing) == 0
    
    return validation_results
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (bool): Whether to fill missing values with column mean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                mean_val = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(mean_val)
                print(f"Filled missing values in column '{col}' with mean: {mean_val:.2f}")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', 'A', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaning data...")
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_strategy='mean', output_path=None):
    """
    Load a CSV file, handle missing values, and optionally save cleaned data.
    
    Parameters:
    file_path (str): Path to the input CSV file.
    fill_strategy (str): Strategy for filling missing values. 
                         Options: 'mean', 'median', 'mode', 'drop', 'zero'.
    output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame if output_path is None, else None.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    original_rows = df.shape[0]
    original_cols = df.shape[1]
    
    if fill_strategy == 'drop':
        df_cleaned = df.dropna()
    elif fill_strategy in ['mean', 'median', 'mode']:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if fill_strategy == 'mean':
            fill_values = df[numeric_cols].mean()
        elif fill_strategy == 'median':
            fill_values = df[numeric_cols].median()
        else:
            fill_values = df[numeric_cols].mode().iloc[0]
        
        df_cleaned = df.copy()
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(fill_values)
        
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        df_cleaned[non_numeric_cols] = df_cleaned[non_numeric_cols].fillna('Unknown')
    elif fill_strategy == 'zero':
        df_cleaned = df.fillna(0)
    else:
        raise ValueError(f"Unsupported fill strategy: {fill_strategy}")
    
    cleaned_rows = df_cleaned.shape[0]
    cleaned_cols = df_cleaned.shape[1]
    
    print(f"Original data: {original_rows} rows, {original_cols} columns")
    print(f"Cleaned data: {cleaned_rows} rows, {cleaned_cols} columns")
    print(f"Missing values handled using: {fill_strategy} strategy")
    
    if output_path:
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return None
    else:
        return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of required column names.
    
    Returns:
    dict: Validation results with status and messages.
    """
    validation_result = {
        'is_valid': True,
        'messages': [],
        'missing_columns': []
    }
    
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['messages'].append('DataFrame is empty')
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_result['is_valid'] = False
            validation_result['missing_columns'] = missing_cols
            validation_result['messages'].append(f'Missing required columns: {missing_cols}')
    
    if df.duplicated().any():
        validation_result['messages'].append(f'Found {df.duplicated().sum()} duplicate rows')
    
    return validation_result

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': ['X', 'Y', np.nan, 'Z', 'W']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_strategy='mean')
    print("\nSample validation:")
    print(validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C']))