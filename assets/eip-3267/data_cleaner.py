
import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing duplicate rows,
    filling missing numeric values with the column median,
    and filling missing categorical values with 'Unknown'.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Handle missing values
    for column in df_cleaned.columns:
        if df_cleaned[column].dtype in [np.float64, np.int64]:
            # Fill numeric columns with median
            median_value = df_cleaned[column].median()
            df_cleaned[column].fillna(median_value, inplace=True)
        else:
            # Fill categorical columns with 'Unknown'
            df_cleaned[column].fillna('Unknown', inplace=True)
    
    return df_cleaned

def validate_dataframe(df):
    """
    Validate that the DataFrame has no missing values after cleaning.
    """
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        return True
    else:
        print(f"DataFrame still has {missing_values} missing values.")
        return False

# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates and missing values
    data = {
        'id': [1, 2, 3, 1, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Eve'],
        'age': [25, 30, np.nan, 25, 35],
        'score': [85.5, 90.0, 78.5, 85.5, np.nan],
        'department': ['HR', 'IT', 'IT', 'HR', None]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\nMissing values in original:")
    print(df.isnull().sum())
    
    # Clean the data
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaning
    is_valid = validate_dataframe(cleaned_df)
    print(f"\nData validation passed: {is_valid}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return dataframe[(dataframe[column] >= lower_bound) & 
                     (dataframe[column] <= upper_bound)]

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using Min-Max scaling.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    result = dataframe.copy()
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            min_val = dataframe[col].min()
            max_val = dataframe[col].max()
            if max_val > min_val:
                result[col] = (dataframe[col] - min_val) / (max_val - min_val)
    
    return result

def standardize_zscore(dataframe, columns=None):
    """
    Standardize specified columns using Z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to standardize (default: all numeric columns)
    
    Returns:
        DataFrame with standardized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    result = dataframe.copy()
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            mean_val = dataframe[col].mean()
            std_val = dataframe[col].std()
            if std_val > 0:
                result[col] = (dataframe[col] - mean_val) / std_val
    
    return result

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        dataframe: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of column names to process (default: all columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns
    
    result = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and dataframe[col].isnull().any():
            if strategy == 'drop':
                result = result.dropna(subset=[col])
            elif strategy == 'mean' and np.issubdtype(dataframe[col].dtype, np.number):
                result[col] = result[col].fillna(dataframe[col].mean())
            elif strategy == 'median' and np.issubdtype(dataframe[col].dtype, np.number):
                result[col] = result[col].fillna(dataframe[col].median())
            elif strategy == 'mode':
                mode_val = dataframe[col].mode()
                if not mode_val.empty:
                    result[col] = result[col].fillna(mode_val[0])
    
    return result

def clean_dataset(dataframe, config=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame
        config: dictionary with cleaning configuration
    
    Returns:
        Cleaned DataFrame
    """
    if config is None:
        config = {
            'missing_strategy': 'mean',
            'outlier_columns': None,
            'outlier_multiplier': 1.5,
            'normalize_columns': None,
            'standardize_columns': None
        }
    
    result = dataframe.copy()
    
    result = handle_missing_values(
        result, 
        strategy=config.get('missing_strategy', 'mean')
    )
    
    outlier_cols = config.get('outlier_columns')
    if outlier_cols:
        for col in outlier_cols:
            if col in result.columns:
                result = remove_outliers_iqr(
                    result, 
                    col, 
                    multiplier=config.get('outlier_multiplier', 1.5)
                )
    
    normalize_cols = config.get('normalize_columns')
    if normalize_cols:
        result = normalize_minmax(result, normalize_cols)
    
    standardize_cols = config.get('standardize_columns')
    if standardize_cols:
        result = standardize_zscore(result, standardize_cols)
    
    return result
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
        keep: {'first', 'last', False} determines which duplicates to keep
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    removed_count = len(df) - len(cleaned_df)
    
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean numeric column by handling missing values.
    
    Args:
        df: pandas DataFrame
        column_name: name of the column to clean
        fill_method: method to fill missing values ('mean', 'median', 'zero')
    
    Returns:
        DataFrame with cleaned column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise ValueError(f"Column '{column_name}' is not numeric")
    
    df_copy = df.copy()
    missing_count = df_copy[column_name].isna().sum()
    
    if missing_count > 0:
        if fill_method == 'mean':
            fill_value = df_copy[column_name].mean()
        elif fill_method == 'median':
            fill_value = df_copy[column_name].median()
        elif fill_method == 'zero':
            fill_value = 0
        else:
            raise ValueError("fill_method must be 'mean', 'median', or 'zero'")
        
        df_copy[column_name] = df_copy[column_name].fillna(fill_value)
        print(f"Filled {missing_count} missing value(s) in '{column_name}' with {fill_method}")
    
    return df_copy

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of column names that must be present
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def get_data_summary(df):
    """
    Generate summary statistics for a DataFrame.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        dict containing summary statistics
    """
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isna().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict(),
        'numeric_columns': list(df.select_dtypes(include=['number']).columns),
        'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns)
    }
    
    return summary