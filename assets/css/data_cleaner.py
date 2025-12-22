
import pandas as pd
import numpy as np

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

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
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
    Normalize a column using specified method.
    
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

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original summary stats:", calculate_summary_stats(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary stats:", calculate_summary_stats(cleaned_df, 'value'))
    
    normalized_df = normalize_column(cleaned_df, 'value', method='zscore')
    print("\nNormalized column added. New columns:", normalized_df.columns.tolist())
    
    return normalized_df

if __name__ == "__main__":
    result_df = example_usage()
    print("\nFirst 5 rows of processed data:")
    print(result_df.head())
import pandas as pd
import numpy as np
from datetime import datetime

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing date columns.
    """
    # Remove duplicate rows
    initial_count = len(df)
    df = df.drop_duplicates()
    removed_duplicates = initial_count - len(df)
    
    # Standardize date columns
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # Format to YYYY-MM-DD if conversion successful
            df[col] = df[col].dt.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"Could not convert column {col}: {e}")
            continue
    
    # Fill missing numeric values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    return df, removed_duplicates

def validate_data(df, required_columns):
    """
    Validate that required columns exist in the DataFrame.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def export_cleaned_data(df, output_path):
    """
    Export cleaned DataFrame to CSV file.
    """
    try:
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(f"Error exporting data: {e}")
        return False

def main():
    # Example usage
    sample_data = {
        'order_date': ['2023-01-15', '2023-01-15', '2023-02-20', None],
        'customer_id': [101, 101, 102, 103],
        'amount': [150.50, 150.50, 200.75, None],
        'product': ['A', 'A', 'B', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df, duplicates_removed = clean_dataframe(df)
    print(f"Removed {duplicates_removed} duplicate rows")
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate required columns
    required_cols = ['order_date', 'customer_id', 'amount']
    try:
        validate_data(cleaned_df, required_cols)
        print("Data validation passed")
    except ValueError as e:
        print(f"Validation error: {e}")
    
    # Export to file
    if export_cleaned_data(cleaned_df, 'cleaned_data.csv'):
        print("Data exported successfully")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Remove outliers using Z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / cleaned_df[numeric_cols].std())
    outlier_mask = (z_scores < outlier_threshold).all(axis=1)
    cleaned_df = cleaned_df[outlier_mask]
    
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns=None, min_rows=10):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if len(df) < min_rows:
        return False, f"Dataset must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset is valid"

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in the dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): Columns to normalize (None for all numeric columns)
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    normalized_df = df.copy()
    
    if columns is None:
        columns = normalized_df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in normalized_df.columns and pd.api.types.is_numeric_dtype(normalized_df[col]):
            if method == 'minmax':
                col_min = normalized_df[col].min()
                col_max = normalized_df[col].max()
                if col_max != col_min:
                    normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            elif method == 'zscore':
                col_mean = normalized_df[col].mean()
                col_std = normalized_df[col].std()
                if col_std != 0:
                    normalized_df[col] = (normalized_df[col] - col_mean) / col_std
    
    return normalized_df

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'feature1': [1, 2, np.nan, 4, 5, 100],
        'feature2': [10, 20, 30, np.nan, 50, 60],
        'category': ['A', 'B', 'A', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned = clean_dataset(df, missing_strategy='mean', outlier_threshold=2)
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned, required_columns=['feature1', 'feature2'])
    print(f"Validation: {is_valid} - {message}")
    print("\n")
    
    # Normalize the numeric columns
    normalized = normalize_data(cleaned, method='minmax')
    print("Normalized DataFrame:")
    print(normalized)
import pandas as pd
import numpy as np
from scipy import stats

def normalize_data(df, columns=None, method='zscore'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
        method: normalization method ('zscore', 'minmax', 'robust')
    
    Returns:
        Normalized DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_normalized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'zscore':
            df_normalized[col] = (df[col] - df[col].mean()) / df[col].std()
        elif method == 'minmax':
            df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        elif method == 'robust':
            median = df[col].median()
            iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
            df_normalized[col] = (df[col] - median) / iqr
    
    return df_normalized

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to check for outliers
        method: outlier detection method ('iqr', 'zscore')
        threshold: threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            mask = z_scores < threshold
        
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def handle_missing_values(df, columns=None, strategy='mean'):
    """
    Handle missing values in specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to handle missing values
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns.tolist()
    
    df_imputed = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'drop':
            df_imputed = df_imputed.dropna(subset=[col])
        elif strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
            df_imputed[col] = df_imputed[col].fillna(df[col].mean())
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
            df_imputed[col] = df_imputed[col].fillna(df[col].median())
        elif strategy == 'mode':
            df_imputed[col] = df_imputed[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
    
    return df_imputed

def clean_dataset(df, config=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        config: dictionary with cleaning configuration
    
    Returns:
        Cleaned DataFrame
    """
    if config is None:
        config = {
            'missing_values': {'strategy': 'mean'},
            'normalization': {'method': 'zscore'},
            'outliers': {'method': 'iqr', 'threshold': 1.5}
        }
    
    df_clean = df.copy()
    
    # Handle missing values
    missing_config = config.get('missing_values', {})
    df_clean = handle_missing_values(df_clean, strategy=missing_config.get('strategy', 'mean'))
    
    # Remove outliers
    outlier_config = config.get('outliers', {})
    df_clean = remove_outliers(
        df_clean, 
        method=outlier_config.get('method', 'iqr'),
        threshold=outlier_config.get('threshold', 1.5)
    )
    
    # Normalize data
    norm_config = config.get('normalization', {})
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    df_clean = normalize_data(
        df_clean, 
        columns=numeric_cols,
        method=norm_config.get('method', 'zscore')
    )
    
    return df_clean