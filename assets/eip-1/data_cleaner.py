
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        data[f'{column}_normalized'] = 0.5
    else:
        data[f'{column}_normalized'] = (data[column] - min_val) / (max_val - min_val)
    
    return data

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        data[f'{column}_standardized'] = 0
    else:
        data[f'{column}_standardized'] = (data[column] - mean_val) / std_val
    
    return data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    data_processed = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if strategy == 'drop':
            data_processed = data_processed.dropna(subset=[col])
        elif strategy == 'mean':
            data_processed[col].fillna(data_processed[col].mean(), inplace=True)
        elif strategy == 'median':
            data_processed[col].fillna(data_processed[col].median(), inplace=True)
        elif strategy == 'mode':
            data_processed[col].fillna(data_processed[col].mode()[0], inplace=True)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return data_processed

def clean_dataset(data, numeric_columns=None, outlier_multiplier=1.5, 
                  normalization_method='minmax', missing_strategy='mean'):
    """
    Comprehensive dataset cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process
        outlier_multiplier: multiplier for IQR outlier detection
        normalization_method: 'minmax' or 'zscore'
        missing_strategy: strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    # Handle missing values
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy, 
                                        columns=numeric_columns)
    
    # Remove outliers
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, col, outlier_multiplier)
    
    # Apply normalization
    for col in numeric_columns:
        if col in cleaned_data.columns:
            if normalization_method == 'minmax':
                cleaned_data = normalize_minmax(cleaned_data, col)
            elif normalization_method == 'zscore':
                cleaned_data = standardize_zscore(cleaned_data, col)
    
    return cleaned_data

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })
    
    print("Original data:")
    print(sample_data)
    print("\nCleaned data:")
    cleaned = clean_dataset(sample_data, numeric_columns=['feature1', 'feature2'])
    print(cleaned)
def remove_duplicates(input_list):
    """
    Remove duplicate items from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_numeric_strings(string_list):
    """
    Convert list of numeric strings to integers, removing non-numeric entries.
    Returns a list of integers.
    """
    cleaned = []
    for s in string_list:
        try:
            cleaned.append(int(s))
        except ValueError:
            continue
    return cleaned

def filter_by_threshold(values, threshold):
    """
    Filter numeric values greater than or equal to threshold.
    Returns a list of filtered values.
    """
    return [v for v in values if v >= threshold]import numpy as np
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
    if max_val == min_val:
        return df[column].apply(lambda x: 0.5)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_cleaning(df_before, df_after, column):
    stats_before = {
        'mean': df_before[column].mean(),
        'std': df_before[column].std(),
        'min': df_before[column].min(),
        'max': df_before[column].max()
    }
    stats_after = {
        'mean': df_after[column].mean(),
        'std': df_after[column].std(),
        'min': df_after[column].min(),
        'max': df_after[column].max()
    }
    return stats_before, stats_after