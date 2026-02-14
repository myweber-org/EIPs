import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Column labels to consider for duplicates
    keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list, optional): List of required column names
    
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
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate a summary of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.to_dict()
    }
    
    return summary
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def handle_missing_values(data, strategy='mean'):
    if strategy == 'mean':
        return data.fillna(data.mean())
    elif strategy == 'median':
        return data.fillna(data.median())
    elif strategy == 'mode':
        return data.fillna(data.mode().iloc[0])
    elif strategy == 'drop':
        return data.dropna()
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'drop'")

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize=False, standardize=False, missing_strategy='mean'):
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            if outlier_method == 'iqr':
                cleaned_data = remove_outliers_iqr(cleaned_data, col)
            elif outlier_method == 'zscore':
                cleaned_data = remove_outliers_zscore(cleaned_data, col)
            
            if normalize:
                cleaned_data = normalize_minmax(cleaned_data, col)
            
            if standardize:
                cleaned_data = standardize_zscore(cleaned_data, col)
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    return cleaned_data

def get_data_summary(data):
    summary = {
        'original_shape': data.shape,
        'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': data.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': data.isnull().sum().to_dict(),
        'data_types': data.dtypes.to_dict()
    }
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.choice(['X', 'Y', 'Z'], 1000),
        'D': np.random.randint(1, 100, 1000)
    })
    
    sample_data.iloc[10:15, 0] = np.nan
    sample_data.iloc[20:25, 1] = np.nan
    
    print("Original data shape:", sample_data.shape)
    summary = get_data_summary(sample_data)
    print("Data summary:", summary)
    
    cleaned = clean_dataset(
        sample_data, 
        numeric_columns=['A', 'B', 'D'],
        outlier_method='iqr',
        normalize=True,
        missing_strategy='mean'
    )
    
    print("Cleaned data shape:", cleaned.shape)
    print("Cleaned data columns:", cleaned.columns.tolist())import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_column(df: pd.DataFrame, column: str, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize specified column using given method.
    
    Args:
        df: Input DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val > min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    return df_copy

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in numeric columns.
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', or 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        df_copy = df_copy.dropna(subset=numeric_cols)
    elif strategy == 'mean':
        for col in numeric_cols:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    elif strategy == 'median':
        for col in numeric_cols:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    return df_copy

def clean_dataset(df: pd.DataFrame, 
                  deduplicate: bool = True,
                  normalize_cols: Optional[List[str]] = None,
                  missing_strategy: str = 'mean') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        normalize_cols: Columns to normalize
        missing_strategy: Strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col)
    
    return cleaned_df
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, msg = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {msg}")
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = 'drop',
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    input_path: Path to input CSV file
    output_path: Path to save cleaned CSV file
    missing_strategy: Strategy for handling missing values ('drop', 'fill', 'interpolate')
    fill_value: Value to fill missing data with when using 'fill' strategy
    
    Returns:
    Cleaned DataFrame
    """
    
    try:
        df = pd.read_csv(input_path)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Handle missing values based on strategy
        if missing_strategy == 'drop':
            df = df.dropna()
        elif missing_strategy == 'fill':
            if fill_value is not None:
                df = df.fillna(fill_value)
            else:
                df = df.fillna(df.mean(numeric_only=True))
        elif missing_strategy == 'interpolate':
            df = df.interpolate(method='linear', limit_direction='forward')
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        
        print(f"Data cleaning completed. Cleaned data saved to {output_path}")
        print(f"Original shape: {pd.read_csv(input_path).shape}, Cleaned shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        raise
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        raise

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame for common data quality issues.
    
    Parameters:
    df: DataFrame to validate
    
    Returns:
    Boolean indicating if data passes validation
    """
    
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    # Check for infinite values
    if np.any(np.isinf(df.select_dtypes(include=[np.number]))):
        print("Validation warning: DataFrame contains infinite values")
    
    # Check for negative values in numeric columns (where negative doesn't make sense)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] < 0).any() and col not in ['temperature_change', 'profit_loss']:
            print(f"Validation warning: Column '{col}' contains negative values")
    
    return True

def calculate_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate basic statistics for numeric columns.
    
    Parameters:
    df: DataFrame to analyze
    
    Returns:
    Dictionary containing statistics
    """
    
    stats = {}
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    for column in numeric_df.columns:
        stats[column] = {
            'mean': float(numeric_df[column].mean()),
            'median': float(numeric_df[column].median()),
            'std': float(numeric_df[column].std()),
            'min': float(numeric_df[column].min()),
            'max': float(numeric_df[column].max()),
            'missing_count': int(df[column].isna().sum())
        }
    
    return stats

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6],
        'value': [10.5, 20.3, np.nan, 15.7, 20.3, 12.1],
        'category': ['A', 'B', 'A', 'B', 'B', 'A']
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data(
        input_path='sample_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='fill',
        fill_value=0.0
    )
    
    if validate_dataframe(cleaned_df):
        stats = calculate_statistics(cleaned_df)
        print("\nData Statistics:")
        for col, col_stats in stats.items():
            print(f"\n{col}:")
            for stat_name, stat_value in col_stats.items():
                print(f"  {stat_name}: {stat_value}")import numpy as np
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
            df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
        else:
            df_copy[f'{column}_normalized'] = 0.5
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val
        else:
            df_copy[f'{column}_normalized'] = 0
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 13, 12, 11, 10]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Remove outliers
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("DataFrame after outlier removal:")
    print(cleaned_df)
    print()
    
    # Calculate statistics
    stats = calculate_summary_statistics(df, 'values')
    print("Summary statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    print()
    
    # Normalize column
    normalized_df = normalize_column(df, 'values', method='minmax')
    print("DataFrame with normalized column:")
    print(normalized_df)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing summary statistics.
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    return stats

if __name__ == "__main__":
    sample_data = {'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 102, 12, 14, 13, 12, 11, 10, 9, 8, 12, 11]}
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print("\nOriginal Statistics:")
    print(calculate_summary_statistics(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Statistics:")
    print(calculate_summary_statistics(cleaned_df, 'values'))
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val - min_val == 0:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    if std_val == 0:
        return df[column]
    return (df[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    required_checks = [
        ('has_nulls', df.isnull().sum().sum() == 0),
        ('has_inf', np.isinf(df.select_dtypes(include=[np.number])).sum().sum() == 0),
        ('has_negative', (df.select_dtypes(include=[np.number]) < 0).sum().sum() == 0)
    ]
    return all(check[1] for check in required_checks)

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['X', 'Y', 'Z'], 200)
    })
    
    print("Original shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b'])
    print("Cleaned shape:", cleaned.shape)
    print("Validation passed:", validate_dataframe(cleaned))
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

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.array([300, 350, -50, 400])
        ])
    }
    
    df = pd.DataFrame(data)
    print(f"Original data shape: {df.shape}")
    print(f"Original statistics: {calculate_summary_statistics(df, 'values')}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print(f"Cleaned statistics: {calculate_summary_statistics(cleaned_df, 'values')}")

if __name__ == "__main__":
    example_usage()
import pandas as pd
import re

def clean_dataset(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict, optional): Dictionary mapping original column names to new names
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns (strip whitespace, lowercase)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
            cleaned_df[col] = cleaned_df[col].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        email_column (str): Name of the column containing email addresses
    
    Returns:
        pd.DataFrame: DataFrame with additional validation columns
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    df['email_valid'] = df[email_column].str.match(email_pattern)
    df['email_domain'] = df[email_column].str.extract(r'@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$')
    
    valid_count = df['email_valid'].sum()
    total_count = len(df)
    
    print(f"Email validation: {valid_count}/{total_count} valid emails ({valid_count/total_count*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', '  BOB JONES  '],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.co.uk'],
        'age': [25, 30, 25, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, normalize_text=True)
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    validated = validate_email_column(cleaned, 'email')
    print("DataFrame with email validation:")
    print(validated)
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to analyze
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        Boolean mask of outliers
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_data = data[(z_scores < threshold) | (data[column].isna())]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].fillna(0)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_data(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].fillna(0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize=False):
    """
    Main function to clean dataset with multiple options.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names to process
        outlier_method: 'iqr' or 'zscore' (default 'iqr')
        normalize: whether to normalize data (default False)
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        # Handle missing values
        cleaned_data[column] = cleaned_data[column].fillna(cleaned_data[column].median())
        
        # Remove outliers
        if outlier_method == 'iqr':
            outliers = detect_outliers_iqr(cleaned_data, column)
            cleaned_data.loc[outliers, column] = cleaned_data[column].median()
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        # Normalize if requested
        if normalize:
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, numeric_columns):
    """
    Validate data quality after cleaning.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names to validate
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {}
    
    for column in numeric_columns:
        if column not in data.columns:
            continue
            
        col_data = data[column]
        validation_results[column] = {
            'missing_values': col_data.isna().sum(),
            'unique_values': col_data.nunique(),
            'mean': col_data.mean(),
            'std': col_data.std(),
            'min': col_data.min(),
            'max': col_data.max(),
            'median': col_data.median()
        }
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000)
    })
    
    # Add some outliers
    sample_data.loc[10, 'feature_a'] = 500
    sample_data.loc[20, 'feature_b'] = 1000
    
    # Clean the data
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    cleaned = clean_dataset(sample_data, numeric_cols, outlier_method='iqr', normalize=True)
    
    # Validate results
    validation = validate_data(cleaned, numeric_cols)
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Validation stats for feature_a: {validation['feature_a']}")