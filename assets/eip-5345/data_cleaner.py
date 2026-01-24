
import pandas as pd

def clean_dataset(df, column_name):
    """
    Clean a DataFrame by removing duplicate rows and sorting by a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_name (str): Column name to sort by.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed and sorted.
    """
    if df.empty:
        return df
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    df_cleaned = df_cleaned.sort_values(by=column_name).reset_index(drop=True)
    
    return df_cleaned

def validate_data(df, required_columns):
    """
    Validate that the DataFrame contains all required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if all required columns are present, False otherwise.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [3, 1, 2, 1, 3, 4],
        'name': ['Charlie', 'Alice', 'Bob', 'Alice', 'Charlie', 'David'],
        'value': [300, 100, 200, 100, 300, 400]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = clean_dataset(df, 'id')
    print("Cleaned DataFrame (sorted by 'id'):")
    print(cleaned_df)
    print()
    
    is_valid = validate_data(cleaned_df, ['id', 'name', 'value'])
    print(f"Data validation passed: {is_valid}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numerical columns
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mean())
    elif strategy == 'median':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
    elif strategy == 'mode':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mode()[0])
    elif strategy == 'drop':
        data = data.dropna(subset=numeric_cols)
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    return data

def validate_dataframe(data):
    """
    Basic DataFrame validation
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    return True

def get_data_summary(data):
    """
    Generate summary statistics for DataFrame
    """
    validate_dataframe(data)
    
    summary = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(data.select_dtypes(include=['object']).columns),
        'missing_values': data.isnull().sum().to_dict(),
        'data_types': data.dtypes.to_dict()
    }
    
    return summaryimport numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data
    
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def clean_dataset(df, numeric_columns):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        df: input DataFrame
        numeric_columns: list of numeric column names
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that required columns exist and have no null values.
    
    Args:
        df: DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing columns: {missing_columns}"
    
    null_counts = df[required_columns].isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0].index.tolist()
    
    if columns_with_nulls:
        return False, f"Columns with null values: {columns_with_nulls}"
    
    return True, "Data validation passed"import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    remove_duplicates (bool): Whether to remove duplicate rows
    fill_method (str): Method to fill missing values - 'mean', 'median', 'mode', or None
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        
        if fill_method == 'mean':
            for col in numeric_cols:
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
        elif fill_method == 'median':
            for col in numeric_cols:
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
        elif fill_method == 'mode':
            for col in cleaned_df.columns:
                cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    else:
        # Drop rows with all NaN values
        cleaned_df.dropna(how='all', inplace=True)
    
    # Remove duplicates
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df.drop_duplicates(inplace=True)
        removed = initial_rows - len(cleaned_df)
        if removed > 0:
            print(f"Removed {removed} duplicate row(s)")
    
    # Reset index after cleaning
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, message)
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

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 3, 4, None],
        'value': [10.5, None, 15.3, 15.3, 20.1, 5.5],
        'category': ['A', 'B', 'A', 'A', 'C', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nShape:", df.shape)
    
    # Clean the data
    cleaned = clean_dataset(df, remove_duplicates=True, fill_method='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nShape:", cleaned.shape)
    
    # Validate
    is_valid, message = validate_dataframe(cleaned, required_columns=['id', 'value'])
    print(f"\nValidation: {message}")