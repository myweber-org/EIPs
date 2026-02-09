
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
    
    return filtered_df.copy()

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
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
        'count': df[column].count()
    }
    
    return stats

def process_numerical_data(df, columns):
    """
    Process multiple numerical columns by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: Processed DataFrame
    """
    processed_df = df.copy()
    
    for col in columns:
        if col in processed_df.columns and pd.api.types.is_numeric_dtype(processed_df[col]):
            processed_df = remove_outliers_iqr(processed_df, col)
    
    return processed_dfimport pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def clean_numeric_data(df, numeric_columns):
    """
    Clean numeric columns by removing outliers and filling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    return cleaned_df

def validate_dataframe(df, required_columns):
    """
    Validate that DataFrame contains required columns and has no empty data.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not all(col in df.columns for col in required_columns):
        return False
    
    if df.empty:
        return False
    
    return True

def process_data_file(file_path, numeric_cols, required_cols):
    """
    Main function to process a data file.
    
    Parameters:
    file_path (str): Path to the data file
    numeric_cols (list): List of numeric column names
    required_cols (list): List of required column names
    
    Returns:
    pd.DataFrame: Processed DataFrame or None if validation fails
    """
    try:
        df = pd.read_csv(file_path)
        
        if not validate_dataframe(df, required_cols):
            print("Data validation failed")
            return None
        
        cleaned_df = clean_numeric_data(df, numeric_cols)
        
        print(f"Original rows: {len(df)}")
        print(f"Cleaned rows: {len(cleaned_df)}")
        print(f"Rows removed: {len(df) - len(cleaned_df)}")
        
        return cleaned_df
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[10:15, 'value'] = 500
    
    numeric_columns = ['value']
    required_columns = ['id', 'value', 'category']
    
    result = process_data_file('sample_data.csv', numeric_columns, required_columns)
    
    if result is not None:
        print("Data processing completed successfully")
        print(result.head())
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    Calculate summary statistics for a column after outlier removal.
    
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
        'count': len(df[column])
    }
    
    return stats

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of summary statistics for each cleaned column
    """
    cleaned_df = df.copy()
    statistics = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            statistics[column] = stats
    
    return cleaned_df, statisticsimport pandas as pd
import numpy as np
from typing import Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[list] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: Input DataFrame
        subset: List of column names to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', fill_value: Optional[float] = None) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: One of 'drop', 'fill', or 'interpolate'
        fill_value: Value to use when strategy is 'fill'
    
    Returns:
        DataFrame with missing values handled
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        if fill_value is not None:
            return df.fillna(fill_value)
        else:
            return df.fillna(df.mean())
    elif strategy == 'interpolate':
        return df.interpolate()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def normalize_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Normalize specified columns to range [0, 1].
    
    Args:
        df: Input DataFrame
        columns: List of column names to normalize
    
    Returns:
        DataFrame with normalized columns
    """
    result = df.copy()
    for col in columns:
        if col in result.columns:
            col_min = result[col].min()
            col_max = result[col].max()
            if col_max != col_min:
                result[col] = (result[col] - col_min) / (col_max - col_min)
    return result

def clean_dataframe(df: pd.DataFrame, 
                    remove_dups: bool = True,
                    missing_strategy: str = 'fill',
                    normalize_cols: Optional[list] = None) -> pd.DataFrame:
    """
    Apply a complete cleaning pipeline to a DataFrame.
    
    Args:
        df: Input DataFrame
        remove_dups: Whether to remove duplicates
        missing_strategy: Strategy for handling missing values
        normalize_cols: List of columns to normalize
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize_cols:
        cleaned_df = normalize_columns(cleaned_df, normalize_cols)
    
    return cleaned_df