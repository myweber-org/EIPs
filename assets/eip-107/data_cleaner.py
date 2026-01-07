import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (bool): Whether to fill missing values. Default is False.
        fill_value: Value to use for filling missing data. Default is 0.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame is valid"

def process_data(file_path, output_path=None):
    """
    Process data from a CSV file, clean it, and optionally save to output.
    
    Args:
        file_path (str): Path to input CSV file.
        output_path (str, optional): Path to save cleaned data.
    
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        
        is_valid, message = validate_dataframe(df)
        if not is_valid:
            raise ValueError(f"Data validation failed: {message}")
        
        cleaned_df = clean_dataframe(df, drop_duplicates=True, fill_missing=True)
        
        if output_path:
            cleaned_df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return cleaned_df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The file is empty")
        raise
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: Optional[str] = None,
    missing_strategy: str = 'mean',
    drop_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and removing problematic columns.
    
    Parameters:
    input_path: Path to input CSV file
    output_path: Optional path to save cleaned data
    missing_strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    drop_threshold: Maximum fraction of missing values allowed per column
    
    Returns:
    Cleaned DataFrame
    """
    
    # Read the CSV file
    df = pd.read_csv(input_path)
    
    # Remove columns with too many missing values
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    # Handle remaining missing values
    for column in df.columns:
        if df[column].isnull().any():
            if missing_strategy == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                df[column].fillna(df[column].mean(), inplace=True)
            elif missing_strategy == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                df[column].fillna(df[column].median(), inplace=True)
            elif missing_strategy == 'mode':
                df[column].fillna(df[column].mode()[0], inplace=True)
            elif missing_strategy == 'drop':
                df = df.dropna(subset=[column])
            else:
                df[column].fillna(df[column].mean(), inplace=True)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    # Save to output path if provided
    if output_path:
        df.to_csv(output_path, index=False)
    
    return df

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate DataFrame and return statistics.
    
    Parameters:
    df: DataFrame to validate
    
    Returns:
    Dictionary with validation statistics
    """
    stats = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'date_columns': len(df.select_dtypes(include=['datetime']).columns)
    }
    
    # Add column-wise statistics
    column_stats = {}
    for column in df.columns:
        column_stats[column] = {
            'dtype': str(df[column].dtype),
            'unique_values': df[column].nunique(),
            'missing_values': df[column].isnull().sum()
        }
        if pd.api.types.is_numeric_dtype(df[column]):
            column_stats[column].update({
                'mean': float(df[column].mean()),
                'std': float(df[column].std()),
                'min': float(df[column].min()),
                'max': float(df[column].max())
            })
    
    stats['column_details'] = column_stats
    return stats

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': ['x', 'y', 'z', 'x', 'y'],
        'D': [1.1, 2.2, 3.3, 4.4, 5.5]
    })
    
    # Save sample data
    sample_data.to_csv('sample_data.csv', index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data('sample_data.csv', 'cleaned_data.csv', missing_strategy='mean')
    
    # Validate the cleaned data
    validation_stats = validate_dataframe(cleaned_df)
    print(f"Cleaned data shape: {cleaned_df.shape}")
    print(f"Validation stats: {validation_stats}")import pandas as pd
import numpy as np

def remove_duplicates(df):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates()

def fill_missing_values(df, strategy='mean'):
    """Fill missing values using specified strategy."""
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        return df.fillna(strategy)

def normalize_column(df, column_name):
    """Normalize specified column to range [0,1]."""
    if column_name in df.columns:
        col = df[column_name]
        df[column_name] = (col - col.min()) / (col.max() - col.min())
    return df

def remove_outliers(df, column_name, threshold=3):
    """Remove outliers using z-score method."""
    if column_name in df.columns:
        z_scores = np.abs((df[column_name] - df[column_name].mean()) / df[column_name].std())
        return df[z_scores < threshold]
    return df

def clean_dataframe(df, operations=None):
    """Apply multiple cleaning operations to DataFrame."""
    if operations is None:
        operations = ['remove_duplicates', 'fill_missing_values']
    
    cleaned_df = df.copy()
    
    for op in operations:
        if op == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df)
        elif op == 'fill_missing_values':
            cleaned_df = fill_missing_values(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, None, 7],
        'B': [10, 20, 20, 40, 50, 60, 1000],
        'C': [100, 200, 300, None, 500, 600, 700]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned)import pandas as pd
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
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: list of columns to fill, if None fills all numeric columns
    
    Returns:
        DataFrame with missing values filled
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if strategy == 'mean':
            fill_value = df[col].mean()
        elif strategy == 'median':
            fill_value = df[col].median()
        elif strategy == 'mode':
            fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
        elif strategy == 'constant':
            fill_value = 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_normalized[col] = (df[col] - mean_val) / std_val
    
    return df_normalized

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a column.
    
    Args:
        df: pandas DataFrame
        column: column name to check for outliers
        method: 'iqr' for interquartile range or 'zscore' for standard deviation
        threshold: multiplier for IQR or cutoff for z-score
    
    Returns:
        Boolean Series indicating outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > threshold
    
    else:
        raise ValueError(f"Unknown method: {method}")

def clean_dataframe(df, operations=None):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df: pandas DataFrame
        operations: list of tuples (operation_name, kwargs)
    
    Returns:
        Cleaned DataFrame
    """
    if operations is None:
        operations = [
            ('remove_duplicates', {}),
            ('fill_missing_values', {'strategy': 'mean'}),
            ('normalize_columns', {'method': 'minmax'})
        ]
    
    cleaned_df = df.copy()
    
    for op_name, kwargs in operations:
        if op_name == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df, **kwargs)
        elif op_name == 'fill_missing_values':
            cleaned_df = fill_missing_values(cleaned_df, **kwargs)
        elif op_name == 'normalize_columns':
            cleaned_df = normalize_columns(cleaned_df, **kwargs)
        else:
            raise ValueError(f"Unknown operation: {op_name}")
    
    return cleaned_df